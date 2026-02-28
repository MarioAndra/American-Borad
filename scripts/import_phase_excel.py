from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.services.excel_import_service import import_directory


def run_import(directory: str, keyword: str = "phase i") -> int:
    """
    Run the Excel import process against the given directory.
    """
    # Basic logging setup suitable for CLI usage
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)

    dir_path = Path(directory)
    if not dir_path.exists() or not dir_path.is_dir():
        logger.error("Directory not found or not a directory: %s", directory)
        return 2

    # Use a single DB session; service will handle transactions per sheet
    db: Session = SessionLocal()
    try:
        summary = import_directory(db, str(dir_path), keyword=keyword)
        # Print machine-readable summary to stdout
        print(json.dumps(summary, default=str, indent=2))
        return 0
    except Exception as e:
        logger.exception("Import failed: %s", e)
        return 1
    finally:
        db.close()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Import Excel files containing 'phase i' data into the database.")
    parser.add_argument("directory", help="Path to the directory containing Excel files (.xlsx/.xls)")
    parser.add_argument(
        "--keyword",
        default="phase i",
        help="Keyword to match in filename or file content (case-insensitive). Default: 'phase i'",
    )
    args = parser.parse_args(argv)
    return run_import(args.directory, args.keyword)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
