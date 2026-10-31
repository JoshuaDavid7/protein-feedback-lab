from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Protein feedback post-training benchmark")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch")
    fetch.add_argument("--data", type=Path, default=Path("data"))
    run = commands.add_parser("run")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--data", type=Path, default=Path("data"))
    for command in ["report", "verify"]:
        sub = commands.add_parser(command)
        sub.add_argument("--output", type=Path, required=True)
        if command == "verify":
            sub.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()
    if args.command == "fetch":
        from .data import fetch_data

        print(fetch_data(args.data))
    elif args.command == "run":
        from .experiment import run_experiment
        from .report import make_report

        run_experiment(json.loads(args.config.read_text()), args.output, args.data)
        make_report(args.output)
    elif args.command == "report":
        from .report import make_report

        make_report(args.output)
    elif args.command == "verify":
        from .report import verify_results

        print(json.dumps(verify_results(args.output, args.data / "raw" / "gb1.csv.zip")))


if __name__ == "__main__":
    main()
