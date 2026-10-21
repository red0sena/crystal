"""Command line: answer one request file.

    python -m crystal examples/request.json [--model DIR_OR_HUB_ID] [--device cuda:0] [--orders 1]

The request is {"state": ..., "questions": {name: {"type": "choice"|"noul", "instructions": ..., "criteria": ...}}}.
"""
import argparse
import json
from pathlib import Path

from .runtime import Crystal

HERE = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("request", type=Path)
    p.add_argument("--model", default=str(HERE), help="model folder or Hub id (default: the folder holding this code)")
    p.add_argument("--device", help="default: cuda:0 when available, else cpu")
    p.add_argument("--orders", type=int, default=1)
    args = p.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    model = Crystal.from_pretrained(args.model, device=args.device, orders=args.orders)
    print(json.dumps(model.system_one(request.get("state"), request["questions"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
