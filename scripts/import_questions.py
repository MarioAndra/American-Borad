from __future__ import annotations

import argparse
import logging
import sys

from app.services.excel_import_service import import_questions_cli


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import questions from an Excel .xlsx file")
    parser.add_argument("excel_path", help="Path to the Excel file (.xlsx)")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    try:
        summary = import_questions_cli(args.excel_path)
        logging.info(
            "Summary: phases=%s topics=%s subtopics=%s abet=%s questions=%s duplicates=%s choices=%s failed_rows=%s",
            summary.phases_created,
            summary.topics_created,
            summary.subtopics_created,
            summary.abet_created,
            summary.questions_created,
            summary.questions_skipped_duplicates,
            summary.choices_created,
            summary.rows_failed,
        )
        return 0
    except Exception as e:
        logging.exception("Import failed: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
