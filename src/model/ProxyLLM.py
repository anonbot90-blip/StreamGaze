"""
ProxyLLM: unified wrapper for closed-source MLLMs routed through the Pluto
LLM proxy (OpenAI-compatible endpoint).

One class, parameterized by `proxy_model`. Supports:
  - GPT-5.4 / GPT-5.x and other reasoning families (max_completion_tokens,
    no temperature)
  - Claude Opus / Sonnet
  - Gemini 2.x-pro, Gemini 3.x (gemini-3 uses reasoning-tokens branch)

The proxy translates OpenAI-format image_url content into the native
per-provider format, so every model family shares the same request shape
here.

Thread-safe: one OpenAI client per process. A per-instance semaphore caps
concurrent outgoing requests at `INNER_THREADS` (env var), so even if the
benchmark loop tries to issue more, the client self-throttles.
"""

from __future__ import annotations

import base64
import io
import os
import random
import threading
import time

import numpy as np
from decord import VideoReader, cpu
from openai import OpenAI, RateLimitError, APIError
from PIL import Image

from model.modelclass import Model


# ---------------------------------------------------------------------------
# Proxy configuration (mirrors agent-reason/core/llm_client.py)
# ---------------------------------------------------------------------------
_DEFAULT_ENDPOINT = os.environ.get(
    "LLM_PROXY_ENDPOINT",
    os.environ.get("OPENAI_BASE_URL", "<model-endpoint>"),
)
_DEFAULT_KEY = os.environ.get(
    "LLM_PROXY_KEY",
    os.environ.get("OPENAI_API_KEY", "sk-REDACTED"),
)

_REASONING_PREFIXES = (
    "o1", "o3", "o4",
    "gpt-5",
    "deepseek-r1",
    "qwen-3.5",
    "gemini-3",
)


def _is_reasoning_model(model_name: str) -> bool:
    m = model_name.lower()
    return any(m.startswith(p) for p in _REASONING_PREFIXES)


class ProxyLLM(Model):
    """
    Generic wrapper for any closed-source model available on the Pluto proxy.

    Args:
        proxy_model: model name on the proxy (e.g. "gpt-5.4",
            "claude-opus-4.6", "gemini-2.5-pro").
        display_name: pretty name returned by .name().
        max_frames: number of video frames per request (default 16, matches
            the paper's closed-source row).
        max_output_tokens: cap on response length (reasoning models use this
            as `max_completion_tokens`; non-reasoning use `max_tokens`).
        max_retries: retry budget for rate-limit / transient errors.
    """

    def __init__(
        self,
        proxy_model: str,
        display_name: str | None = None,
        max_frames: int = 16,
        max_output_tokens: int = 1536,
        max_retries: int = 5,
    ):
        self.proxy_model = proxy_model
        self.display_name = display_name or proxy_model
        self.max_frames = max_frames
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        self.is_reasoning = _is_reasoning_model(proxy_model)

        # If reasoning, be more generous — reasoning tokens are counted.
        if self.is_reasoning and max_output_tokens < 4096:
            self.max_output_tokens = 4096

        # Inner-thread ceiling for self-throttling. Capped at whatever the
        # script sets; default 1 (pure serial, same as original GPT4o.py).
        inner = int(os.environ.get("INNER_THREADS", "1"))
        self._semaphore = threading.Semaphore(max(1, inner))

        endpoint = _DEFAULT_ENDPOINT.rstrip("/")
        base_url = endpoint if endpoint.endswith("/v1") else endpoint + "/v1"
        self.client = OpenAI(base_url=base_url, api_key=_DEFAULT_KEY)

        print(
            f"🔧 ProxyLLM initialized: model='{self.proxy_model}' "
            f"reasoning={self.is_reasoning} max_frames={self.max_frames} "
            f"inner_threads={inner} endpoint={base_url}"
        )

    # ------------------------------------------------------------------
    # Benchmark contract
    # ------------------------------------------------------------------
    def Run(
        self,
        file,
        inp,
        start_time,
        end_time,
        question_time,
        omni=False,
        proactive=False,
        salience_map_path=None,
    ):
        return self._run(file, inp, start_time, end_time, salience_map_path)

    def name(self):
        return self.display_name

    # ------------------------------------------------------------------
    # Video frame extraction — matches GPT4o.py behavior
    # ------------------------------------------------------------------
    def _extract_frames(self, video_path: str, start_time: float, end_time: float):
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            fps = vr.get_avg_fps()
            start_frame = int(max(0, start_time) * fps)
            end_frame = int(max(start_frame + 1, end_time * fps))
            end_frame = min(end_frame, len(vr))

            total = end_frame - start_frame
            if total <= 0:
                return None

            if total > self.max_frames:
                indices = np.linspace(start_frame, end_frame - 1, self.max_frames, dtype=int)
            else:
                indices = list(range(start_frame, end_frame))

            frames = vr.get_batch(indices).asnumpy()
            return frames
        except Exception as e:
            print(f"❌ ProxyLLM frame extraction failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Image encode — matches GPT4o.py: max 2048 px, JPEG Q=85, base64
    # ------------------------------------------------------------------
    @staticmethod
    def _encode_image_b64(frame_array) -> str | None:
        try:
            img = Image.fromarray(frame_array.astype("uint8"), "RGB")
            max_side = 2048
            if max(img.size) > max_side:
                ratio = max_side / max(img.size)
                new_size = tuple(int(d * ratio) for d in img.size)
                img = img.resize(new_size, Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as e:
            print(f"❌ ProxyLLM image encode failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Single chat.completions.create call with retry
    # ------------------------------------------------------------------
    def _call_api(self, messages):
        kwargs = {"model": self.proxy_model, "messages": messages}

        if self.is_reasoning:
            kwargs["max_completion_tokens"] = self.max_output_tokens
            # Deliberately omit temperature / top_p — some reasoning models reject them.
        else:
            kwargs["max_tokens"] = self.max_output_tokens
            kwargs["temperature"] = 0.7
            # Intentionally omit top_p: Bedrock-backed Claude (Opus 4.6) rejects
            # requests that specify BOTH temperature and top_p. Keeping only
            # temperature is compatible with all three providers.

        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                return resp
            except RateLimitError as e:
                last_err = e
                # Exponential backoff with jitter — jitter avoids thundering herd
                # when many parallel workers retry simultaneously.
                delay = (2 ** attempt) + random.uniform(0, 1.0)
                if attempt < self.max_retries - 1:
                    print(
                        f"⚠️  [{self.proxy_model}] Rate limit "
                        f"(attempt {attempt + 1}/{self.max_retries}), "
                        f"sleeping {delay:.1f}s"
                    )
                    time.sleep(delay)
                else:
                    print(f"❌ [{self.proxy_model}] Rate limit after {self.max_retries} attempts")
                    raise
            except APIError as e:
                last_err = e
                delay = (2 ** attempt) + random.uniform(0, 0.5)
                if attempt < self.max_retries - 1:
                    print(
                        f"⚠️  [{self.proxy_model}] API error "
                        f"(attempt {attempt + 1}/{self.max_retries}): {e}, "
                        f"sleeping {delay:.1f}s"
                    )
                    time.sleep(delay)
                else:
                    print(f"❌ [{self.proxy_model}] API error after {self.max_retries} attempts")
                    raise
            except Exception as e:  # pragma: no cover
                last_err = e
                delay = (2 ** attempt) + random.uniform(0, 0.5)
                if attempt < self.max_retries - 1:
                    print(
                        f"⚠️  [{self.proxy_model}] Unexpected error: {e} "
                        f"(attempt {attempt + 1}/{self.max_retries}), "
                        f"sleeping {delay:.1f}s"
                    )
                    time.sleep(delay)
                else:
                    raise

        raise RuntimeError(f"[{self.proxy_model}] exhausted retries: {last_err}")

    # ------------------------------------------------------------------
    # Core run
    # ------------------------------------------------------------------
    def _run(self, file, inp, start_time, end_time, salience_map_path):
        try:
            t0 = time.time()
            duration = max(0.0, end_time - start_time)
            print(
                f"📹 [{self.proxy_model}] {start_time:.1f}s → {end_time:.1f}s "
                f"({duration:.1f}s, {self.max_frames} frames)"
            )

            frames = self._extract_frames(file, start_time, end_time)
            if frames is None or len(frames) == 0:
                print(f"❌ [{self.proxy_model}] frame extraction returned empty")
                return " ", 0

            content = []
            for frame in frames:
                b64 = self._encode_image_b64(frame)
                if b64:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    })

            if salience_map_path and os.path.exists(salience_map_path):
                try:
                    with open(salience_map_path, "rb") as f:
                        sb64 = base64.b64encode(f.read()).decode("utf-8")
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{sb64}"},
                    })
                except Exception as e:
                    print(f"⚠️  salience map load failed: {e}")

            content.append({"type": "text", "text": inp})
            messages = [{"role": "user", "content": content}]

            # Self-throttle — cap in-flight requests per process to INNER_THREADS.
            with self._semaphore:
                response = self._call_api(messages)

            output = (response.choices[0].message.content or "").strip()
            latency = time.time() - t0

            # tiny courtesy pause between requests per-thread (matches GPT4o.py)
            time.sleep(0.1)

            preview = output.replace("\n", " ")[:120]
            print(f"💬 [{self.proxy_model}] → {preview!r}  ({latency:.2f}s)")
            return output, latency

        except Exception as e:
            import traceback
            print(f"❌ [{self.proxy_model}] inference error: {e}")
            traceback.print_exc()
            return " ", 0
