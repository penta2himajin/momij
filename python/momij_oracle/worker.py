#!/usr/bin/env python3
"""JSONL worker: exact Maple generate/benchmark via mlx-lm-deepgrove."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Callable


class PhaseStats:
    """Accumulates wall times for decode (single-token) phases with forced mx.eval."""

    def __init__(self) -> None:
        self.decode_moe = 0
        self.decode_attn = 0
        self.router_s = 0.0
        self.switch_s = 0.0
        self.agg_s = 0.0
        self.attn_s = 0.0
        self.prefill_moe = 0
        self.prefill_moe_s = 0.0

    def as_ms_avg(self) -> dict[str, float]:
        n_moe = max(self.decode_moe, 1)
        n_attn = max(self.decode_attn, 1)
        return {
            "decode_moe_calls": float(self.decode_moe),
            "decode_attn_calls": float(self.decode_attn),
            "router_ms": 1e3 * self.router_s / n_moe,
            "switch_ms": 1e3 * self.switch_s / n_moe,
            "agg_ms": 1e3 * self.agg_s / n_moe,
            "moe_ms": 1e3 * (self.router_s + self.switch_s + self.agg_s) / n_moe,
            "attn_ms": 1e3 * self.attn_s / n_attn,
            "prefill_moe_calls": float(self.prefill_moe),
            "prefill_moe_ms": (
                1e3 * self.prefill_moe_s / self.prefill_moe if self.prefill_moe else 0.0
            ),
        }


def _is_decode(x, hidden: int) -> bool:
    # Prefill is [B,L,H] with L>1; decode is often [B,1,H] or flat H.
    try:
        return int(x.size) == hidden
    except Exception:
        return False


def install_sync_profile(model, stats: PhaseStats) -> Callable[[], None]:
    """Wrap Maple mlp/attn modules (instance __call__ patch is ignored by Python)."""
    import mlx.core as mx
    from mlx_lm.models.maple import aggregate_expert_outputs

    hidden = int(model.args.hidden_size)
    restores: list[tuple[Any, str, Any]] = []

    class _Wrap:
        def __init__(self, inner, call):
            object.__setattr__(self, "_inner", inner)
            object.__setattr__(self, "_call", call)

        def __call__(self, *args, **kwargs):
            return self._call(self._inner, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    for layer in model.model.layers:
        mlp = layer.mlp
        if hasattr(mlp, "gate") and hasattr(mlp, "switch_mlp"):

            def moe_call(inner, x, _h=hidden):
                decode = _is_decode(x, _h)
                t0 = time.perf_counter()
                inds, scores = inner.gate(x)
                mx.eval(inds, scores)
                t1 = time.perf_counter()
                y = inner.switch_mlp(x, inds)
                mx.eval(y)
                t2 = time.perf_counter()
                out = aggregate_expert_outputs(y, scores)
                mx.eval(out)
                t3 = time.perf_counter()
                if decode:
                    stats.decode_moe += 1
                    stats.router_s += t1 - t0
                    stats.switch_s += t2 - t1
                    stats.agg_s += t3 - t2
                else:
                    stats.prefill_moe += 1
                    stats.prefill_moe_s += t3 - t0
                return out

            restores.append((layer, "mlp", mlp))
            layer.mlp = _Wrap(mlp, moe_call)

        attn = layer.self_attn

        def attn_call(inner, x, mask=None, cache=None, _h=hidden):
            decode = _is_decode(x, _h)
            t0 = time.perf_counter()
            y = inner(x, mask, cache)
            mx.eval(y)
            t1 = time.perf_counter()
            if decode:
                stats.decode_attn += 1
                stats.attn_s += t1 - t0
            return y

        restores.append((layer, "self_attn", attn))
        layer.self_attn = _Wrap(attn, attn_call)

    def uninstall() -> None:
        for obj, name, prev in restores:
            setattr(obj, name, prev)

    return uninstall


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--flash-head", action="store_true")
    args = ap.parse_args()

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm import stream_generate
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler

    model_config = {"use_flash_head": True} if args.flash_head else None
    model, tokenizer = load(
        args.model, lazy=False, model_config=model_config, trust_remote_code=True
    )
    print(json.dumps({"status": "ready"}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            cmd = req.get("cmd")
            if cmd == "generate":
                prompt = req["prompt"]
                max_tokens = int(req.get("max_tokens", 256))
                temp = float(req.get("temperature", 0))
                sampler = make_sampler(temp=temp) if temp > 0 else make_sampler(temp=0)
                tokens = []
                for resp in stream_generate(
                    model,
                    tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    sampler=sampler,
                ):
                    tok = getattr(resp, "token", None)
                    if tok is None:
                        continue
                    tokens.append(int(tok))
                print(json.dumps({"tokens": tokens}), flush=True)
            elif cmd == "benchmark":
                p = int(req.get("prompt_tokens", 128))
                g = int(req.get("generation_tokens", 256))
                n = int(req.get("num_trials", 3))
                want_profile = bool(req.get("profile")) or os.environ.get(
                    "MOMIJ_PROFILE_MOE"
                ) == "1"

                prompt = mx.array([[100] * p])
                for _ in generate_step(prompt[0], model, max_tokens=8):
                    pass
                mx.metal.clear_cache()
                ptps, gtps, peaks = [], [], []
                for _ in range(n):
                    prompt = mx.array([[100] * p])
                    mx.eval(prompt)
                    tic = time.time()
                    y = []
                    t_prefill_end = None
                    for i, (token, logprobs) in enumerate(
                        generate_step(prompt[0], model, max_tokens=g)
                    ):
                        if i == 0:
                            t_prefill_end = time.time()
                        y.append(token)
                        if len(y) >= g:
                            break
                    toc = time.time()
                    prefill_s = (t_prefill_end or tic) - tic
                    gen_s = toc - (t_prefill_end or tic)
                    ptps.append(p / max(prefill_s, 1e-6))
                    gtps.append(g / max(gen_s, 1e-6))
                    peaks.append(mx.metal.get_peak_memory() / 1e9)

                out: dict[str, Any] = {
                    "prompt_tps": sum(ptps) / len(ptps),
                    "generation_tps": sum(gtps) / len(gtps),
                    "peak_memory": max(peaks) if peaks else 0,
                }

                if want_profile:
                    stats = PhaseStats()
                    uninstall = install_sync_profile(model, stats)
                    try:
                        prompt = mx.array([[100] * p])
                        mx.eval(prompt)
                        # One instrumented run; phase times use forced eval (sync).
                        for i, _ in enumerate(
                            generate_step(prompt[0], model, max_tokens=g)
                        ):
                            if i + 1 >= g:
                                break
                    finally:
                        uninstall()
                    prof = stats.as_ms_avg()
                    out["profile"] = prof
                    sys.stderr.write(
                        "[moe-profile] oracle sync "
                        f"decode_moe={int(prof['decode_moe_calls'])} "
                        f"router_ms={prof['router_ms']:.3f} "
                        f"switch_ms={prof['switch_ms']:.3f} "
                        f"agg_ms={prof['agg_ms']:.3f} "
                        f"moe_ms={prof['moe_ms']:.3f} "
                        f"attn_ms={prof['attn_ms']:.3f}\n"
                    )
                    sys.stderr.flush()

                print(json.dumps(out), flush=True)
            else:
                print(json.dumps({"error": f"unknown cmd {cmd}"}), flush=True)
        except Exception as e:
            print(json.dumps({"error": str(e)}), flush=True)


if __name__ == "__main__":
    main()
