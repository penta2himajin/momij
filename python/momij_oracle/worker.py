#!/usr/bin/env python3
"""JSONL worker: exact Maple generate/benchmark via mlx-lm-deepgrove."""
from __future__ import annotations

import argparse
import json
import sys
import time


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
                    # stream_generate yields GenerationResponse with .token
                    tok = getattr(resp, "token", None)
                    if tok is None:
                        continue
                    tokens.append(int(tok))
                print(json.dumps({"tokens": tokens}), flush=True)
            elif cmd == "benchmark":
                # Replicate mlx_lm.benchmark timing shape with random-ish ids.
                import mlx_lm.benchmark as bench  # noqa: F401

                p = int(req.get("prompt_tokens", 128))
                g = int(req.get("generation_tokens", 256))
                n = int(req.get("num_trials", 3))
                prompt = mx.array([[100] * p])
                # warmup
                for _ in generate_step(prompt[0], model, max_tokens=8):
                    pass
                mx.metal.clear_cache()
                ptps, gtps, peaks = [], [], []
                for _ in range(n):
                    prompt = mx.array([[100] * p])
                    mx.eval(prompt)
                    tic = time.time()
                    # prefill measured inside first generate_step
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
                    # approximate: first token includes prefill
                    prefill_s = (t_prefill_end or tic) - tic
                    gen_s = toc - (t_prefill_end or tic)
                    ptps.append(p / max(prefill_s, 1e-6))
                    gtps.append(g / max(gen_s, 1e-6))
                    peaks.append(mx.metal.get_peak_memory() / 1e9)
                print(
                    json.dumps(
                        {
                            "prompt_tps": sum(ptps) / len(ptps),
                            "generation_tps": sum(gtps) / len(gtps),
                            "peak_memory": max(peaks) if peaks else 0,
                        }
                    ),
                    flush=True,
                )
            else:
                print(json.dumps({"error": f"unknown cmd {cmd}"}), flush=True)
        except Exception as e:
            print(json.dumps({"error": str(e)}), flush=True)


if __name__ == "__main__":
    main()
