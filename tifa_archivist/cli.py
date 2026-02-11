from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys

from .config import DEFAULT_CONFIG_PATH, load_config
from .db import ImageDB
from .logging_config import setup_logging
from .orchestrator import run_pipeline
from .quality_audit import run_quality_audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tifa_archivist",
        description="Search, download, classify, and sort Tifa Lockhart images.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to config.yaml",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run", help="Search, download, classify, and store images."
    )
    run_parser.add_argument("--limit", type=int, default=None, help="Max images to keep")
    run_parser.add_argument("--out", type=Path, default=None, help="Output directory")
    run_parser.add_argument(
        "--manifest",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Write manifest.json",
    )
    run_parser.add_argument(
        "--skip-classify",
        action="store_true",
        help="Skip xAI classification and save all images to other/",
    )

    stats_parser = subparsers.add_parser("stats", help="Show dataset counts")
    stats_parser.add_argument("--out", type=Path, default=None, help="Output directory")

    audit_parser = subparsers.add_parser(
        "audit", help="Audit downloaded images for gray/low-quality outputs."
    )
    audit_parser.add_argument("--out", type=Path, default=None, help="Output directory")
    audit_parser.add_argument(
        "--limit", type=int, default=200, help="Max images to audit"
    )
    audit_parser.add_argument(
        "--llm",
        action="store_true",
        help="Use Gemini to label a sample of flagged images",
    )
    audit_parser.add_argument(
        "--llm-limit", type=int, default=20, help="Max images sent to LLM"
    )
    audit_parser.add_argument(
        "--report",
        type=Path,
        default=Path("QUALITYREPORT.MD"),
        help="Report output path",
    )
    audit_parser.add_argument(
        "--json",
        type=Path,
        default=Path("quality_report.json"),
        help="JSON output path",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "run":
        if args.limit is not None:
            config.limit = args.limit
        if args.out is not None:
            config.out_dir = args.out
            config.db_path = config.out_dir / "images.db"
        if args.manifest is not None:
            config.manifest = args.manifest
        if args.skip_classify:
            config.skip_classify = True

        setup_logging(config.log_dir)
        try:
            asyncio.run(run_pipeline(config))
        except KeyboardInterrupt:
            print("Interrupted.", file=sys.stderr)
    elif args.command == "stats":
        out_dir = args.out or config.out_dir
        db = ImageDB(out_dir / "images.db")
        db.init()
        counts = db.count_by_category()
        total = sum(counts.values())
        print(f"Total records: {total}")
        for category, count in sorted(counts.items()):
            print(f"{category}: {count}")
    elif args.command == "audit":
        out_dir = args.out or config.out_dir
        if args.out is not None:
            config.out_dir = args.out
            config.db_path = config.out_dir / "images.db"
        run_quality_audit(
            config=config,
            out_dir=out_dir,
            limit=args.limit,
            report_path=args.report,
            json_path=args.json,
            llm=args.llm,
            llm_limit=args.llm_limit,
        )
