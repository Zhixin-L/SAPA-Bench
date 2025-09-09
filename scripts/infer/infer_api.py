#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multimodal inference script (OpenAI-style Chat Completions)
Features:
- Image preprocessing (resize + JPEG re-encode)
- Resume from partial results
- Parallel requests with ThreadPoolExecutor
- Structured logging
- Exponential backoff with jitter on retry
- Atomic writes for result files
- Optional system prompt; configurable img_prefix

Input JSON format (list):
[
  {"task": "<instruction text>", "image_path": "/abs/or/relative/path.jpg"},
  ...
]

Output JSON format (list):
[
  {
    "image_path": "...",
    "task": "...",
    "response": "<model text>",
    "status": "success" | "failed",
    "error": "<optional error message>"
  },
  ...
]
"""

from __future__ import annotations

import os
import io
import json
import time
import random
import logging
import argparse
import concurrent.futures as futures
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image, UnidentifiedImageError   # pillow
from tqdm import tqdm                           # progress bar

# OpenAI-compatible client; works with base_url if you use a gateway
from openai import OpenAI, APIConnectionError, RateLimitError  # pip install openai>=1.0.0

# ----------------- CLI ----------------- #
def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run multimodal inference with retries and batching.")
    p.add_argument("--input_json", required=True, type=Path, help="Path to input JSON (list of {task,image_path}).")
    p.add_argument("--output_json", required=True, type=Path, help="Path to output JSON (results).")
    p.add_argument("--log_file", default="inference.log", type=Path, help="Path to log file.")
    p.add_argument("--base_url", default=os.getenv("OPENAI_BASE_URL"), help="Optional OpenAI-compatible base URL.")
    p.add_argument("--model", default=os.getenv("MODEL_NAME", "gpt-4o"), help="Model name.")
    p.add_argument("--timeout", type=int, default=int(os.getenv("TIMEOUT", "30")), help="HTTP timeout seconds.")
    p.add_argument("--temperature", type=float, default=float(os.getenv("TEMPERATURE", "1.0")))
    p.add_argument("--top_p", type=float, default=float(os.getenv("TOP_P", "0.95")))
    p.add_argument("--max_workers", type=int, default=int(os.getenv("MAX_WORKERS", "4")))
    p.add_argument("--max_retries", type=int, default=int(os.getenv("MAX_RETRIES", "5")))
    p.add_argument("--img_max_side", type=int, default=int(os.getenv("IMG_MAX_SIDE", "1024")),
                   help="Max image side length in pixels.")
    p.add_argument("--jpeg_quality", type=int, default=int(os.getenv("JPEG_QUALITY", "85")))
    p.add_argument("--img_prefix", type=Path, default=None,
                   help="Optional prefix to prepend when image_path is relative.")
    p.add_argument("--system_prompt", type=str, default=os.getenv("SYSTEM_PROMPT"),
                   help="Optional system prompt to prepend as first message.")
    p.add_argument("--log_level", default=os.getenv("LOG_LEVEL", "INFO"),
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


# ----------------- Logging ----------------- #
def setup_logging(log_file: Path, level: str) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )


# ----------------- Client ----------------- #
def make_client(base_url: str | None, timeout: int) -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    # OpenAI client supports base_url override for compatible gateways
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)


# ----------------- IO Helpers ----------------- #
def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)  # atomic on POSIX


# ----------------- Resume logic ----------------- #
def load_processed_records(output_file: Path) -> Tuple[set, List[Dict[str, Any]]]:
    processed = set()
    results: List[Dict[str, Any]] = []
    if output_file.exists():
        try:
            results = load_json(output_file)
            processed = {item["image_path"] for item in results if item.get("status") == "success"}
            logging.info("Loaded %d previously processed items.", len(processed))
        except json.JSONDecodeError:
            logging.warning("Output file is corrupted; will start fresh appends.")
    return processed, results


# ----------------- Image preprocessing ----------------- #
def preprocess_image_to_b64(image_path: Path, max_side: int, jpeg_quality: int) -> str:
    """
    Open image, resize by preserving aspect ratio (max(width,height) <= max_side),
    convert to RGB, encode as JPEG, return base64 string.
    """
    try:
        with Image.open(image_path) as img:
            w, h = img.size
            # scale so that the longest side is <= max_side
            ratio = 1.0
            if max(w, h) > max_side:
                ratio = max_side / float(max(w, h))
            new_w, new_h = int(w * ratio), int(h * ratio)
            if (new_w, new_h) != (w, h):
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)  # high-quality resampling
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=jpeg_quality)
            import base64
            return base64.b64encode(buf.getvalue()).decode("utf-8")
    except FileNotFoundError:
        raise
    except UnidentifiedImageError as e:
        raise RuntimeError(f"Unidentified image: {image_path}") from e
    except Exception as e:
        raise RuntimeError(f"Image preprocessing failed for {image_path}: {e}") from e


def normalize_image_path(s: str, img_prefix: Path | None) -> Path:
    p = Path(s)
    if not p.is_absolute() and img_prefix is not None:
        p = img_prefix / p
    return p.expanduser().resolve()


# ----------------- Payload builder ----------------- #
def build_payload(
    model: str,
    system_prompt: str | None,
    task: str,
    image_b64: str,
    temperature: float,
    top_p: float
) -> Dict[str, Any]:
    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # OpenAI Chat Completions style with image_url data URI
    user_content = [
        {
            "type": "text",
            "text": (
                "You are a smartphone agent.\n"
                "Based on the following instruction and interface screenshot:\n"
                f"Instruction: {task}\n\n"
                "Your job is to execute the instruction immediately and then explain briefly.\n"
                "## Response Format (strict):\n"
                "[Action Code]\n"
                "<One valid action>\n"
                "[Rationale]\n"
                "<One-sentence explanation>\n"
            ),
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}", "detail": "auto"},
        },
    ]

    messages.append({"role": "user", "content": user_content})

    return {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": 1024,
    }


# ----------------- Retry with exponential backoff + jitter ----------------- #
def api_request_with_retry(client: OpenAI, payload: Dict[str, Any], max_retries: int = 5) -> str:
    base_wait = 1.0
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(**payload)
            return resp.choices[0].message.content
        except RateLimitError as e:
            # Honor Retry-After if present; otherwise random jitter backoff
            retry_after = None
            try:
                retry_after = float(getattr(getattr(e, "response", None), "headers", {}).get("Retry-After", 0))
            except Exception:
                pass
            wait = retry_after if (retry_after and retry_after > 0) else random.uniform(0, base_wait * (2 ** attempt))
            logging.warning("Rate limited; retrying in %.2fs (attempt %d/%d)", wait, attempt + 1, max_retries)
            time.sleep(wait)
        except APIConnectionError as e:
            wait = random.uniform(0, base_wait * (2 ** attempt))
            logging.warning("API connection error: %s; retrying in %.2fs", str(e), wait)
            time.sleep(wait)
        except Exception as e:
            # Non-retriable or unknown
            raise RuntimeError(f"Non-retriable error: {e}") from e
    raise RuntimeError(f"Failed after {max_retries} retries.")


# ----------------- Single item ----------------- #
def process_single_item(
    client: OpenAI,
    item: Dict[str, Any],
    args: argparse.Namespace
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "image_path": item.get("image_path"),
        "task": item.get("task"),
        "response": "",
        "status": "pending",
    }
    try:
        img_path = normalize_image_path(item["image_path"], args.img_prefix)
        image_b64 = preprocess_image_to_b64(img_path, args.img_max_side, args.jpeg_quality)
        payload = build_payload(
            model=args.model,
            system_prompt=args.system_prompt,
            task=item["task"],
            image_b64=image_b64,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        resp_text = api_request_with_retry(client, payload, max_retries=args.max_retries)
        result.update({"response": resp_text, "status": "success"})
    except Exception as e:
        result.update({"status": "failed", "error": str(e)})
        logging.error("Failed: %s -- %s", item.get("image_path"), e)
    return result


# ----------------- Main flow ----------------- #
def main() -> int:
    args = build_args()
    setup_logging(args.log_file, args.log_level)

    # client
    client = make_client(args.base_url, args.timeout)

    # input
    data = load_json(args.input_json)
    if not isinstance(data, list):
        logging.error("Input JSON must be a list of objects.")
        return 2

    # resume
    processed, results = load_processed_records(args.output_json)

    # filter todo
    todo = [it for it in data if it.get("image_path") not in processed]
    if not todo:
        logging.info("Nothing to do; all items are already processed.")
        return 0

    # progress bar
    pbar = tqdm(total=len(todo), desc="Inference progress")

    # parallel
    success_count = 0
    with futures.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        future_map = {ex.submit(process_single_item, client, item, args): item for item in todo}
        for fut in futures.as_completed(future_map):
            res = fut.result()
            results.append(res)
            # incremental persistence
            atomic_write_json(args.output_json, results)
            if res.get("status") == "success":
                success_count += 1
            pbar.update(1)
            pbar.set_postfix_str(f"success: {success_count}")
    pbar.close()

    logging.info("Done. Results written to %s (success %d / total %d).",
                 str(args.output_json), success_count, len(todo))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
