#!/usr/bin/env python
"""SERL / GaussianActor pure-model inference benchmark.

Measures per-step inference latency and theoretical max control frequency.
Supports CPU, CUDA, and Ascend NPU via ``--device``.
"""

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file
from torch import Tensor

from lerobot.configs import parser
from lerobot.policies import make_policy
from lerobot.rl.train_rl import TrainRLServerPipelineConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BENCHMARK_ARGS: argparse.Namespace | None = None


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------
def _synchronize_device(device: torch.device) -> None:
    """Block until all work on *device* is complete (for wall-clock timing)."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "npu":
        torch.npu.synchronize()


# ---------------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------------
class TimingBuffer:
    """Collect per-step wall-clock measurements and compute summary statistics."""

    def __init__(self) -> None:
        self._data: dict[str, list[float]] = {}

    def record(self, key: str, value: float) -> None:
        self._data.setdefault(key, []).append(value)

    def stats(self, key: str) -> dict[str, float]:
        arr = np.array(self._data.get(key, []))
        if arr.size == 0:
            return {}
        return {
            "mean_ms": float(np.mean(arr)) * 1000,
            "p50_ms": float(np.median(arr)) * 1000,
            "p95_ms": float(np.percentile(arr, 95)) * 1000,
            "p99_ms": float(np.percentile(arr, 99)) * 1000,
            "max_ms": float(np.max(arr)) * 1000,
            "mean_hz": float(1.0 / np.mean(arr)),
            "p95_hz": float(1.0 / np.percentile(arr, 95)),
        }

    def all_stats(self) -> dict[str, dict[str, float]]:
        return {k: self.stats(k) for k in self._data}


# ---------------------------------------------------------------------------
# Dummy observation
# ---------------------------------------------------------------------------
def _make_dummy_observation(policy) -> dict[str, Tensor]:
    """Build a single FP32 dummy observation matching the policy input spec."""
    device = next(policy.parameters()).device
    obs: dict[str, Tensor] = {}
    for key, feature in policy.config.input_features.items():
        shape = tuple(feature.shape)
        obs[key] = torch.randn(1, *shape, device=device, dtype=torch.float32)
    return obs


# ---------------------------------------------------------------------------
# Benchmark: full-step timing
# ---------------------------------------------------------------------------
def _benchmark_inference(policy, obs: dict, n_iters: int, n_warmup: int) -> TimingBuffer:
    """Time end-to-end ``select_action`` calls with device synchronisation."""
    device = next(policy.parameters()).device
    timings = TimingBuffer()

    for _ in range(n_warmup):
        with torch.inference_mode():
            _ = policy.select_action(obs)

    for _ in range(n_iters):
        _synchronize_device(device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            _ = policy.select_action(obs)
        _synchronize_device(device)
        timings.record("inference", time.perf_counter() - t0)

    return timings


# ---------------------------------------------------------------------------
# Benchmark: phase breakdown (best-effort, wall-clock only)
# ---------------------------------------------------------------------------
def _benchmark_phases(policy, obs: dict, n_iters: int, timings: TimingBuffer) -> None:
    """Time individual pipeline stages on a subset of iterations.

    Notes
    -----
    Phase timings use wall-clock ``time.perf_counter`` *without* device
    synchronisation between stages, so they are approximate and should be
    interpreted as rough proportions rather than exact kernel durations.
    """
    n_phase = min(n_iters, 100)

    for _ in range(n_phase):
        # -- image encoder cache ------------------------------------------------
        t0 = time.perf_counter()
        with torch.inference_mode():
            cache = None
            if getattr(policy.actor.encoder, "has_images", False):
                cache = policy.actor.encoder.get_cached_image_features(obs)
        t1 = time.perf_counter()

        # -- full encoder forward -----------------------------------------------
        with torch.inference_mode():
            obs_enc = policy.actor.encoder(obs, cache=cache)
        t2 = time.perf_counter()

        # -- actor MLP ----------------------------------------------------------
        with torch.inference_mode():
            _ = policy.actor.network(obs_enc)
        t3 = time.perf_counter()

        # -- discrete critic (optional) -----------------------------------------
        if getattr(policy, "discrete_critic", None) is not None:
            with torch.inference_mode():
                _ = policy.discrete_critic(obs, observation_features=cache)
            t4 = time.perf_counter()
            timings.record("discrete_critic", t4 - t3)

        timings.record("image_encoder", t1 - t0)
        timings.record("state_encoder", t2 - t1)
        timings.record("actor_mlp", t3 - t2)


# ---------------------------------------------------------------------------
# Benchmark entry-point
# ---------------------------------------------------------------------------
def benchmark_pure_model(
    policy, n_iters: int = 200, n_warmup: int = 10
) -> dict[str, Any]:
    """Run the pure-model benchmark and return a summary dict."""
    policy.eval()
    device = next(policy.parameters()).device
    obs = _make_dummy_observation(policy)

    timings = _benchmark_inference(policy, obs, n_iters, n_warmup)
    _benchmark_phases(policy, obs, n_iters, timings)

    stats = timings.all_stats()
    return {
        "mode": "pure_model",
        "n_iters": n_iters,
        "device": str(device),
        "stats": stats,
        "max_control_freq_mean_hz": stats.get("inference", {}).get("mean_hz", 0.0),
        "max_control_freq_p95_hz": stats.get("inference", {}).get("p95_hz", 0.0),
    }


# ---------------------------------------------------------------------------
# Checkpoint loading with key remapping
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


def _load_and_remap_state_dict(policy_path: str) -> dict[str, torch.Tensor]:
    model_file = Path(policy_path) / "model.safetensors"
    if not model_file.exists():
        raise FileNotFoundError(f"No model.safetensors found in {policy_path}")
    state_dict = load_file(str(model_file))
    return {_remap_state_dict_key(k): v for k, v in state_dict.items()}


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------
def run_benchmark(
    cfg: TrainRLServerPipelineConfig, *, pure_model: bool, n_iters: int
) -> dict[str, Any]:
    """Build the policy, load weights, and run benchmarks."""
    cfg.validate()

    policy_path = None
    if cfg.policy is not None and cfg.policy.pretrained_path is not None:
        policy_path = str(cfg.policy.pretrained_path)
        cfg.policy.pretrained_path = None

    logging.disable(logging.WARNING)
    policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env)
    logging.disable(logging.NOTSET)
    policy.eval()

    if policy_path:
        logger.info(f"Loading checkpoint from {policy_path} with key remapping...")
        state_dict = _load_and_remap_state_dict(policy_path)
        model_keys = set(policy.state_dict().keys())
        filtered = {k: v for k, v in state_dict.items() if k in model_keys}
        model_dtype = next(policy.parameters()).dtype
        filtered = {k: v.to(model_dtype) for k, v in filtered.items()}
        missing, unexpected = policy.load_state_dict(filtered, strict=False)
        ignored = ("encoder_actor.", "encoder_critic.", "discrete_critic.encoder.")
        real_missing = [k for k in missing if not k.startswith(ignored)]
        if real_missing:
            logger.warning(f"Missing key(s): {real_missing}")
        if unexpected:
            logger.warning(f"Unexpected key(s): {unexpected}")
        if not real_missing and not unexpected:
            logger.info("Checkpoint loaded cleanly after remap.")

    if pure_model:
        return benchmark_pure_model(policy, n_iters=n_iters)
    raise NotImplementedError("Only --pure_model is supported in this minimal version.")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def print_report(result: dict[str, Any]) -> None:
    stats = result.get("stats", {})
    print("\n" + "=" * 70)
    print(" SERL INFERENCE BENCHMARK REPORT")
    print("=" * 70)
    print(f"Mode:            {result['mode']}")
    print(f"Device:          {result['device']}")
    print(f"Iterations:      {result['n_iters']}")
    print("\n--- Timing Statistics (ms) ---")
    for name, s in stats.items():
        print(
            f"{name:20s} mean={s['mean_ms']:7.2f}  p95={s['p95_ms']:7.2f}  "
            f"max={s['max_ms']:7.2f}  (mean_hz={s['mean_hz']:7.1f}, p95_hz={s['p95_hz']:7.1f})"
        )
    print("\n--- Max Control Frequency ---")
    print(f"Based on mean step time: {result['max_control_freq_mean_hz']:.1f} Hz")
    print(f"Based on p95 step time:  {result['max_control_freq_p95_hz']:.1f} Hz")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_benchmark_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--pure_model", action="store_true")
    ap.add_argument("--n_iters", type=int, default=200)
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)
    args, remaining = ap.parse_known_args(argv[1:])

    # Map our --device shortcut to LeRobot's --policy.device so draccus
    # handles the parsing natively.
    if args.device:
        remaining = [f"--policy.device={args.device}", *remaining]

    sys.argv = [argv[0], *remaining]
    return args


@parser.wrap()
def main(cfg: TrainRLServerPipelineConfig) -> None:
    global BENCHMARK_ARGS
    if BENCHMARK_ARGS is None:
        BENCHMARK_ARGS = _parse_benchmark_args(sys.argv)

    if not BENCHMARK_ARGS.pure_model:
        raise SystemExit("This minimal benchmark only supports --pure_model mode.")

    result = run_benchmark(cfg, pure_model=True, n_iters=BENCHMARK_ARGS.n_iters)
    print_report(result)

    if BENCHMARK_ARGS.output_dir:
        output_dir = Path(BENCHMARK_ARGS.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "benchmark_report.json"
        with open(report_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Report saved to {report_path}")


if __name__ == "__main__":
    BENCHMARK_ARGS = _parse_benchmark_args(sys.argv)
    main()
