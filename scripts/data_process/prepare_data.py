#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def convert_a_to_b(a_path: Path, b_path: Path, img_prefix: Path) -> int:
    if not a_path.is_file():
        logger.error("Input file does not exist: %s", a_path)
        return 1

    b_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        data_a: List[Dict[str, Any]] = json.loads(a_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.exception("Failed to read or parse input JSON: %s", e)
        return 2

    data_b = []
    for i, item in enumerate(data_a):
        instructions = (item.get("conversations1") or {}).get("instructions", "")
        image_name = item.get("images", "")
        entry = {
            "task": instructions,
            "image_path": str(img_prefix / image_name)
        }
        data_b.append(entry)

    try:
        b_path.write_text(json.dumps(data_b, ensure_ascii=False, indent=4), encoding="utf-8")
    except Exception as e:
        logger.exception("Failed to write output JSON: %s", e)
        return 3

    logger.info("Created file: %s with %d entries", b_path, len(data_b))
    return 0

def main() -> int:
    parser = argparse.ArgumentParser(description="Convert input JSON to task + image_path format.")
    parser.add_argument("--a_path", type=Path, required=True, help="Path to input JSON file (a.json)")
    parser.add_argument("--b_path", type=Path, required=True, help="Path to output JSON file (b.json)")
    parser.add_argument("--img_prefix", type=Path, default=Path("/hy-tmp/Positive_sample/Positive sample/"),
                        help="Prefix to prepend to each image filename")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level (default: INFO)")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(levelname)s: %(message)s")

    return convert_a_to_b(args.a_path, args.b_path, args.img_prefix)

if __name__ == "__main__":
    sys.exit(main())
