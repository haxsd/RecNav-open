#!/usr/bin/env python
"""Small OpenAI connectivity smoke test."""

from __future__ import annotations

import argparse
import os
import sys
import time

from ada_semnav.llm_client import default_model_for_provider, infer_provider, resolve_client_kwargs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify OpenAI API connectivity.")
    parser.add_argument(
        "--provider",
        choices=["auto", "openai", "deepseek"],
        default="auto",
        help="LLM provider to test.",
    )
    parser.add_argument("--model", default="", help="Model name (empty = provider default).")
    parser.add_argument("--timeout-sec", type=float, default=20.0, help="Request timeout seconds.")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("python package 'openai' is not installed in current environment.") from exc

    provider = infer_provider() if args.provider == "auto" else args.provider
    if provider == "none":
        raise RuntimeError("No API key found. Set OPENAI_API_KEY or DEEPSEEK_API_KEY in .env.")

    model = args.model.strip() or default_model_for_provider(provider=provider)
    kwargs = resolve_client_kwargs(provider=provider)
    if not kwargs.get("api_key", ""):
        raise RuntimeError(f"API key missing for provider={provider}.")

    client = OpenAI(timeout=args.timeout_sec, **kwargs)
    t0 = time.perf_counter()
    text = ""
    try:
        resp = client.responses.create(
            model=model,
            input=[{"role": "user", "content": "Reply only with: OK"}],
            max_output_tokens=16,
        )
        text = (getattr(resp, "output_text", "") or "").strip()
    except Exception:
        chat = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply only with: OK"}],
            temperature=0.0,
            max_tokens=16,
        )
        if getattr(chat, "choices", None):
            text = (chat.choices[0].message.content or "").strip()
    t1 = time.perf_counter()
    print("LLM smoke finished.")
    print(f"- provider: {provider}")
    print(f"- model: {model}")
    print(f"- latency_ms: {(t1 - t0) * 1000.0:.2f}")
    print(f"- output: {text[:120]}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
