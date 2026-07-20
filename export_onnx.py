#!/usr/bin/env python
"""Export a SERL GaussianActor checkpoint to ONNX and verify correctness.

The exported model is deterministic — it outputs (means, stds, discrete_logits)
instead of sampled actions.  Sampling is done in post-processing:

    eps = np.random.randn(batch, 3)
    continuous = np.tanh(means + stds * eps)
    discrete = np.argmax(discrete_logits, axis=-1, keepdims=True)
    action = np.concatenate([continuous, discrete], axis=-1)

Usage::

    # Basic export (dynamic batch)
    python export_onnx.py \\
        --config_path sac_gym_hil_pick_lift/eval_rl_config.json \\
        --policy.path=sac_gym_hil_pick_lift/eval_policy_pretrained \\
        --output policy.onnx

    # Export with verification
    python export_onnx.py \\
        --config_path sac_gym_hil_pick_lift/eval_rl_config.json \\
        --policy.path=sac_gym_hil_pick_lift/eval_policy_pretrained \\
        --output policy.onnx --verify

    # Fixed batch=1 (for ATC / Ascend NPU deployment)
    python export_onnx.py \\
        --config_path sac_gym_hil_pick_lift/eval_rl_config.json \\
        --policy.path=sac_gym_hil_pick_lift/eval_policy_pretrained \\
        --output policy_bs1.onnx --fixed_batch 1

    # FP16 image encoder for Ascend NPU
    python export_onnx.py \\
        --config_path sac_gym_hil_pick_lift/eval_rl_config.json \\
        --policy.path=sac_gym_hil_pick_lift/eval_policy_pretrained \\
        --output policy_fp16.onnx --dtype fp16

Ascend NPU deployment workflow::

    # 1. Export ONNX with fixed batch size
    python export_onnx.py ... --fixed_batch 1 --output policy_bs1.onnx

    # 2. ONNX → OM via ATC (Ascend Tensor Compiler)
    atc --model=policy_bs1.onnx --output=policy_bs1 \\
        --framework=5 --soc_version=Ascend310B

    # 3. Run inference with ACL / pyacl / MindSpore Lite
"""

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
from safetensors.torch import load_file
from torch import Tensor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NPU workaround (safe to call even without NPU present)
# ---------------------------------------------------------------------------
dist.Distribution.set_default_validate_args(False)
if getattr(torch, "npu", None) is not None:
    torch.npu.config.allow_internal_format = False

# Lazy imports to keep the script self-contained for the non-lerobot path
from lerobot.configs import parser as lerobot_parser
from lerobot.configs.parser import get_path_arg
from lerobot.policies import make_policy
from lerobot.rl.train_rl import TrainRLServerPipelineConfig
from lerobot.configs.policies import PreTrainedConfig


# ---------------------------------------------------------------------------
# FP16 image-encoder wrapper (same as benchmark_serl_inference.py)
# ---------------------------------------------------------------------------
class FloatInHalfOutImageEncoder(nn.Module):
    """Wrap an image encoder so it runs internally in FP16 but exposes FP32.

    On Ascend 310B the pooling kernel ``MaxPoolWithArgmaxV1`` only supports
    FP16, so the image encoder must run in half-precision.
    """

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder.half()

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x.half()).float()


# ---------------------------------------------------------------------------
# Checkpoint loading helpers (mirrors benchmark_serl_inference.py)
# ---------------------------------------------------------------------------
def _remap_state_dict_key(key: str) -> str:
    key = key.replace("_orig_mod.", "")
    patterns = [
        (
            r"^actor\.encoder\.image_encoder\.image_enc_layers\.embedder\.1\.(weight|bias)$",
            r"actor.encoder.image_encoder.image_enc_layers.embedder.1.group_norm.\1",
        ),
        (
            r"^actor\.encoder\.image_encoder\.image_enc_layers\.encoder\.stages\.(\d+)\.norm([12])\.(weight|bias)$",
            r"actor.encoder.image_encoder.image_enc_layers.encoder.stages.\1.norm\2.group_norm.\3",
        ),
        (
            r"^actor\.encoder\.image_encoder\.image_enc_layers\.encoder\.stages\.(\d+)\.shortcut\.1\.(weight|bias)$",
            r"actor.encoder.image_encoder.image_enc_layers.encoder.stages.\1.shortcut.1.group_norm.\2",
        ),
    ]
    for pat, repl in patterns:
        if re.match(pat, key):
            return re.sub(pat, repl, key)
    return key


def _load_and_remap_state_dict(checkpoint_dir: str) -> dict[str, Tensor]:
    model_file = Path(checkpoint_dir) / "model.safetensors"
    if not model_file.exists():
        raise FileNotFoundError(f"No model.safetensors found in {checkpoint_dir}")
    raw = load_file(str(model_file))
    return {_remap_state_dict_key(k): v for k, v in raw.items()}


def build_wrapper_and_export(
    checkpoint_dir: str,
    output_path: str,
    *,
    cfg: TrainRLServerPipelineConfig,
    use_fp16: bool = False,
    verify: bool = False,
    opset: int = 17,
    fixed_batch: int | None = None,
    use_dynamo: bool = True,
) -> None:
    """Core export logic: load policy, build wrapper, export, verify.

    *cfg* is the parsed pipeline config already validated.
    """
    cfg.validate()

    # When the pipeline config has no policy section, load the policy config
    # directly from the checkpoint's config.json.
    if cfg.policy is None:
        import json

        from lerobot.policies.gaussian_actor.configuration_gaussian_actor import GaussianActorConfig

        policy_config_path = Path(checkpoint_dir) / "config.json"
        with open(policy_config_path) as f:
            policy_config_dict = json.load(f)
        policy_config = GaussianActorConfig(**policy_config_dict)
        policy_config.pretrained_path = None
    else:
        policy_config = cfg.policy
        if policy_config.pretrained_path is not None:
            policy_config.pretrained_path = None

    # Force CPU for ONNX export (TorchScript tracer requires CPU tensors).
    policy_config.device = "cpu"

    logging.disable(logging.WARNING)
    policy = make_policy(cfg=policy_config, env_cfg=cfg.env)
    logging.disable(logging.NOTSET)
    policy.eval()

    # Load weights with key remapping
    logger.info(f"Loading checkpoint from {checkpoint_dir} ...")
    state_dict = _load_and_remap_state_dict(checkpoint_dir)
    model_keys = set(policy.state_dict().keys())
    filtered = {k: v for k, v in state_dict.items() if k in model_keys}
    model_dtype = next(policy.parameters()).dtype
    filtered = {k: v.to(model_dtype) for k, v in filtered.items()}
    missing, unexpected = policy.load_state_dict(filtered, strict=False)

    ignored_prefixes = ("encoder_actor.", "encoder_critic.", "discrete_critic.encoder.")
    real_missing = [k for k in missing if not k.startswith(ignored_prefixes)]
    if real_missing:
        logger.warning(f"Missing keys: {real_missing}")
    if not real_missing and not unexpected:
        logger.info("Checkpoint loaded cleanly after remap.")
    else:
        logger.info(f"Loaded with {len(real_missing)} missing, {len(unexpected)} unexpected keys (expected).")

    # Optional FP16 image encoder for NPU
    if use_fp16:
        logger.info("Converting image encoder to FP16 (for Ascend NPU compatibility).")
        enc = policy.actor.encoder
        if hasattr(enc, "image_encoder") and enc.image_encoder is not None:
            enc.image_encoder = FloatInHalfOutImageEncoder(enc.image_encoder)
        disc = getattr(policy, "discrete_critic", None)
        if disc is not None and hasattr(disc.encoder, "image_encoder"):
            if disc.encoder.image_encoder is not None:
                disc.encoder.image_encoder = FloatInHalfOutImageEncoder(disc.encoder.image_encoder)

    # Build ONNX wrapper (moves modules from policy → wrapper)
    logger.info("Building ONNX export wrapper ...")
    wrapper = OnnxExportWrapper(policy)

    # Export
    export_onnx(wrapper, output_path, opset=opset, fixed_batch=fixed_batch, use_dynamo=use_dynamo)

    # Verify
    if verify:
        logger.info("\n--- Verification ---")
        result = verify_onnx(wrapper, output_path)
        if result["passed"]:
            logger.info("Verification PASSED: all outputs match within tolerance.")
        else:
            logger.error("Verification FAILED: some outputs differ beyond tolerance.")
            raise SystemExit(1)

    logger.info("\nDone. Model exported to: %s", output_path)
    logger.info("Inputs:  front_img [B,3,128,128], wrist_img [B,3,128,128], state_vector [B,18]")
    logger.info("Outputs: means [B,3], stds [B,3], discrete_logits [B,3]")
    logger.info("\nPost-processing for sampling:")
    logger.info("  eps = np.random.randn(batch, 3)")
    logger.info("  continuous = np.tanh(means + stds * eps)")
    logger.info("  discrete = np.argmax(discrete_logits, axis=-1, keepdims=True)")
    logger.info("  action = np.concatenate([continuous, discrete], axis=-1)")


# ---------------------------------------------------------------------------
# ONNX export wrapper
# ---------------------------------------------------------------------------
class OnnxExportWrapper(nn.Module):
    """Deterministic wrapper that outputs (means, stds, discrete_logits).

    Built by moving sub-modules out of a loaded GaussianActorPolicy so that
    parameters are tracked in the wrapper's own module hierarchy.  The
    original policy object **must not be used** after this wrapper is built.
    """

    def __init__(self, policy) -> None:
        super().__init__()
        actor = policy.actor
        enc = actor.encoder

        # Image encoder (shared across front / wrist views)
        self.add_module("image_encoder", enc.image_encoder)
        # Spatial embeddings + post-encoders per camera.
        # Keys in the ModuleDict use '.' → '_' replacement (see _init_image_layers).
        front_key = "observation.images.front".replace(".", "_")
        wrist_key = "observation.images.wrist".replace(".", "_")
        self.add_module("spatial_front", enc.spatial_embeddings[front_key])
        self.add_module("spatial_wrist", enc.spatial_embeddings[wrist_key])
        self.add_module("post_front", enc.post_encoders[front_key])
        self.add_module("post_wrist", enc.post_encoders[wrist_key])
        # State encoder
        self.add_module("state_enc", enc.state_encoder)
        # Actor MLP + heads
        self.add_module("actor_mlp", actor.network)
        self.add_module("mean_layer", actor.mean_layer)
        self.add_module("std_layer", actor.std_layer)

        # Discrete critic (optional)
        disc = getattr(policy, "discrete_critic", None)
        if disc is not None:
            self.add_module("disc_mlp", disc.net)
            self.add_module("disc_head", disc.output_layer)
            self._has_discrete = True
        else:
            self._has_discrete = False

        self._std_min = 1e-5
        self._std_max = 5.0

    def _encode_image(self, img: Tensor, spatial: nn.Module, post: nn.Module) -> Tensor:
        feats: Tensor = self.image_encoder(img)  # [B, 512, 4, 4]
        pooled: Tensor = spatial(feats)           # [B, 4096]
        return post(pooled)                       # [B, 64]

    def forward(
        self, front_img: Tensor, wrist_img: Tensor, state_vector: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        # --- image encoding --------------------------------------------------
        front_feat = self._encode_image(front_img, self.spatial_front, self.post_front)
        wrist_feat = self._encode_image(wrist_img, self.spatial_wrist, self.post_wrist)

        # --- state encoding --------------------------------------------------
        state_feat: Tensor = self.state_enc(state_vector)

        # --- shared latent ---------------------------------------------------
        latent = torch.cat([front_feat, wrist_feat, state_feat], dim=-1)  # [B, 192]

        # --- actor head ------------------------------------------------------
        actor_out = self.actor_mlp(latent)
        means: Tensor = self.mean_layer(actor_out)
        stds: Tensor = torch.exp(self.std_layer(actor_out)).clamp(self._std_min, self._std_max)

        # --- discrete critic -------------------------------------------------
        if self._has_discrete:
            disc_out = self.disc_mlp(latent)
            disc_logits: Tensor = self.disc_head(disc_out)
        else:
            disc_logits = torch.zeros(means.shape[0], 0, device=means.device)

        return means, stds, disc_logits


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------
def export_onnx(
    wrapper: OnnxExportWrapper,
    output_path: str,
    opset: int = 17,
    fixed_batch: int | None = None,
    use_dynamo: bool = True,
) -> None:
    """Export *wrapper* to ONNX.

    When *fixed_batch* is set (e.g. 1), the output has a static batch dimension
    suitable for ATC / CANN toolchain conversion.  Otherwise a dynamic "batch"
    dimension is emitted.

    *use_dynamo* uses the new torch.export-based ONNX exporter (PyTorch >= 2.9).
    Set to False to fall back to the legacy TorchScript-based exporter.
    """
    wrapper.eval()

    # Dummy inputs (batch = 1)
    front = torch.randn(1, 3, 128, 128)
    wrist = torch.randn(1, 3, 128, 128)
    state = torch.randn(1, 18)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting ONNX model (this may take a minute for the first run)...")

    if use_dynamo:
        # New torch.export-based path (PyTorch >= 2.9 default).
        # Uses dynamic_shapes instead of dynamic_axes.
        if fixed_batch is not None:
            dynamic_shapes = None
        else:
            batch_dim = torch.export.Dim("batch")
            dynamic_shapes = {
                "front_img": {0: batch_dim},
                "wrist_img": {0: batch_dim},
                "state_vector": {0: batch_dim},
            }

        torch.onnx.export(
            wrapper,
            (front, wrist, state),
            output_path,
            input_names=["front_img", "wrist_img", "state_vector"],
            output_names=["means", "stds", "discrete_logits"],
            dynamic_shapes=dynamic_shapes,
            opset_version=opset,
        )
    else:
        # Legacy TorchScript-based path.
        if fixed_batch is not None:
            dynamic_axes = {}
        else:
            dynamic_axes = {
                "front_img": {0: "batch"},
                "wrist_img": {0: "batch"},
                "state_vector": {0: "batch"},
                "means": {0: "batch"},
                "stds": {0: "batch"},
                "discrete_logits": {0: "batch"},
            }

        torch.onnx.export(
            wrapper,
            (front, wrist, state),
            output_path,
            input_names=["front_img", "wrist_img", "state_vector"],
            output_names=["means", "stds", "discrete_logits"],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            export_params=True,
            dynamo=False,
        )

    # Dynamo exporter writes large weights to an external .onnx.data file.
    # Merge them back into a single-file model for easier deployment (ATC etc.).
    data_file = Path(str(output_path) + ".data")
    if data_file.exists():
        import onnx

        logger.info("Merging external data into single-file ONNX model ...")
        model = onnx.load(output_path)
        onnx.save_model(model, output_path, save_as_external_data=False)
        data_file.unlink()
        logger.info("External data merged and cleaned up.")

    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    logger.info(f"Exported to {output_path} ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Verification: PyTorch vs ONNX Runtime
# ---------------------------------------------------------------------------
def verify_onnx(
    wrapper: OnnxExportWrapper,
    onnx_path: str,
    batch_sizes: tuple[int, ...] = (1, 4),
    atol: float = 1e-4,
) -> dict[str, Any]:
    """Compare PyTorch and ONNX Runtime outputs."""
    import onnxruntime as ort

    wrapper.eval()
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    # Detect the model's batch dimension (fixed or dynamic).
    input_shape = session.get_inputs()[0].shape
    model_batch = input_shape[0] if isinstance(input_shape[0], int) else None

    results: dict[str, Any] = {"passed": True, "outputs": {}}

    for bs in batch_sizes:
        if model_batch is not None and bs != model_batch:
            continue  # skip batch sizes that don't match the fixed model
        torch.manual_seed(42)
        front = torch.randn(bs, 3, 128, 128)
        wrist = torch.randn(bs, 3, 128, 128)
        state = torch.randn(bs, 18)

        # PyTorch
        with torch.inference_mode():
            pt_means, pt_stds, pt_logits = wrapper(front, wrist, state)

        # ONNX Runtime
        ort_inputs = {
            "front_img": front.numpy().astype(np.float32),
            "wrist_img": wrist.numpy().astype(np.float32),
            "state_vector": state.numpy().astype(np.float32),
        }
        ort_means, ort_stds, ort_logits = session.run(None, ort_inputs)

        # Compare
        for name, pt_tensor, ort_arr in [
            ("means", pt_means, ort_means),
            ("stds", pt_stds, ort_stds),
            ("discrete_logits", pt_logits, ort_logits),
        ]:
            pt_np = pt_tensor.numpy()
            max_diff = float(np.abs(pt_np - ort_arr).max())
            results["outputs"][f"{name}_bs{bs}"] = max_diff

            status = "OK" if max_diff < atol else "FAIL"
            logger.info(f"  [{status}] {name:20s}  batch={bs}  max_diff={max_diff:.2e}")
            if max_diff >= atol:
                results["passed"] = False

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
EXPORT_ARGS: argparse.Namespace | None = None


def _parse_export_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--output", type=str, default=None,
                    help="Output ONNX path")
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--dtype", type=str, choices=["fp32", "fp16"], default="fp32",
                    help="Image encoder dtype: fp32 (default) or fp16 for Ascend NPU")
    ap.add_argument("--fixed_batch", type=int, default=None,
                    help="Export with fixed batch size (for ATC / CANN toolchain)")
    ap.add_argument("--legacy", action="store_true",
                    help="Use legacy TorchScript-based exporter (default: dynamo-based)")
    ap.add_argument("--verify", action="store_true")
    args, remaining = ap.parse_known_args(argv[1:])

    sys.argv = [argv[0], *remaining]
    return args


@lerobot_parser.wrap()
def main(cfg: TrainRLServerPipelineConfig) -> None:
    global EXPORT_ARGS
    if EXPORT_ARGS is None:
        EXPORT_ARGS = _parse_export_args(sys.argv)

    # 'policy.path' is a special key handled by the lerobot parser — it is
    # extracted from CLI args and config JSON, then available via get_path_arg.
    checkpoint = get_path_arg("policy")

    output_path = EXPORT_ARGS.output or str(Path(checkpoint).parent / "policy.onnx")

    build_wrapper_and_export(
        str(checkpoint),
        output_path,
        cfg=cfg,
        use_fp16=(EXPORT_ARGS.dtype == "fp16"),
        verify=EXPORT_ARGS.verify,
        opset=EXPORT_ARGS.opset,
        fixed_batch=EXPORT_ARGS.fixed_batch,
        use_dynamo=not EXPORT_ARGS.legacy,
    )


if __name__ == "__main__":
    EXPORT_ARGS = _parse_export_args(sys.argv)
    main()
