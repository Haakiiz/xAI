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
        "audit",
        help="Audit kept images for visual quality (gray/flat/blurry detection).",
    )
    audit_parser.add_argument("--out", type=Path, default=None, help="Output directory")
    audit_parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Max images to audit (most recent first)",
    )
    audit_parser.add_argument(
        "--report",
        type=Path,
        default=Path("QUALITYREPORT.MD"),
        help="Markdown report path",
    )
    audit_parser.add_argument(
        "--json",
        type=Path,
        default=Path("quality_report.json"),
        help="JSON report path (full per-image metrics)",
    )
    audit_parser.add_argument(
        "--llm",
        action="store_true",
        help="Additionally call Gemini on flagged/top-N images (needs GEMINI_API_KEY)",
    )
    audit_parser.add_argument(
        "--llm-limit",
        type=int,
        default=30,
        help="Max images to send to Gemini when --llm is set",
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
        setup_logging(config.log_dir)
        run_quality_audit(
            config,
            out_dir,
            args.limit,
            args.report,
            args.json,
            args.llm,
            args.llm_limit,
        )
        print(f"Wrote {args.report} and {args.json}")
