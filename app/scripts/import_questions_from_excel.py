import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.db.session import SessionLocal
from app.models import (
    Phase,
    Topic,
    SubTopic,
    Question,
    Choice,
    ABETCriterion,
    DifficultyLevel,
)
from app.models.enums import cognitive_level_from_value

DIFFICULTY_MAP = {
    "Easy": DifficultyLevel.Easy,
    "Medium": DifficultyLevel.Medium,
    "Hard": DifficultyLevel.Hard,
}


def get_or_create(db: Session, model, defaults=None, **kwargs):
    instance = db.execute(
        select(model).filter_by(**kwargs)
    ).scalar_one_or_none()

    if instance:
        return instance

    instance = model(**kwargs, **(defaults or {}))
    db.add(instance)
    db.flush()
    return instance


def import_excel(file_path: str, created_by_user_id: int | None = None):
    df = pd.read_excel(file_path, engine="openpyxl")
    def norm(s: str) -> str:
        return "".join(ch if ch.isalnum() else " " for ch in s.lower()).split()
    def map_col(s: str) -> str:
        tokens = norm(s)
        j = " ".join(tokens)
        if j == "phase":
            return "phase"
        if j == "topic":
            return "topic"
        if j == "subtopic":
            return "subtopic"
        if j == "cognitive level":
            return "cognitive_level"
        if j == "difficulty":
            return "difficulty"
        if j == "question type":
            return "question_type"
        if j == "question text":
            return "question_text"
        if j == "abet outcomes":
            return "abet_outcomes"
        if j == "correct answer" or j == "answer":
            return "correct_answer"
        if j in {"option a", "a"}:
            return "Option A"
        if j in {"option b", "b"}:
            return "Option B"
        if j in {"option c", "c"}:
            return "Option C"
        if j in {"option d", "d"}:
            return "Option D"
        return s.strip()
    df = df.rename(columns={c: map_col(str(c)) for c in df.columns})
    if "phase" not in df.columns or "topic" not in df.columns or "subtopic" not in df.columns:
        df = pd.read_excel(file_path, engine="openpyxl", header=1)
        df = df.rename(columns={c: map_col(str(c)) for c in df.columns})
    db = SessionLocal()

    try:
        for idx, row in df.iterrows():

            # -------- Phase --------
            phase = get_or_create(
                db,
                Phase,
                name=str(row["phase"]).strip()
            )

            # -------- Topic --------
            topic = get_or_create(
                db,
                Topic,
                phase_id=phase.id,
                name=str(row["topic"]).strip()
            )

            # -------- Subtopic --------
            subtopic = get_or_create(
                db,
                SubTopic,
                topic_id=topic.id,
                name=str(row["subtopic"]).strip()
            )

            # -------- ABET --------
            abet = None
            if pd.notna(row["abet_outcomes"]):
                abet = get_or_create(
                    db,
                    ABETCriterion,
                    code=str(row["abet_outcomes"]).strip(),
                    defaults={"name": str(row["abet_outcomes"]).strip()}
                )

            # -------- ENUM mapping --------
            difficulty_raw = str(row["difficulty"]).strip()
            if difficulty_raw not in DIFFICULTY_MAP:
                raise ValueError(f"Invalid difficulty at row {idx + 2}")
            difficulty = DIFFICULTY_MAP[difficulty_raw]

            cognitive = cognitive_level_from_value(str(row["cognitive_level"]))

            qtype_raw = str(row["question_type"]).strip()
            qtype = qtype_raw

            # -------- Question --------
            question = get_or_create(
                db,
                Question,
                text=str(row["question_text"]).strip(),
                defaults={
                    "difficulty": difficulty,
                    "cognitive_level": cognitive,
                    "question_type": qtype,
                    "subtopic_id": subtopic.id,
                    "abet_criterion_id": abet.id if abet else None,
                    "explanation": str(row.get("explanation", "")).strip(),
                    "common_mistake": str(row.get("common_mistake", "")).strip(),
                    "skill_gap": str(row.get("skill_gap", "")).strip(),
                    "created_by": created_by_user_id,
                }
            )

            # -------- Choices --------
            correct = str(row["correct_answer"]).strip().upper()

            options = {
                "A": row["Option A"],
                "B": row["Option B"],
                "C": row["Option C"],
                "D": row["Option D"],
            }

            for key, text in options.items():
                if pd.isna(text):
                    continue

                get_or_create(
                    db,
                    Choice,
                    question_id=question.id,
                    text=str(text).strip(),
                    defaults={"is_correct": key == correct}
                )

        db.commit()
        print("✅ Excel file imported successfully")

    except Exception as e:
        db.rollback()
        print("❌ Import failed:", e)
        raise

    finally:
        db.close()


def _main():
    import argparse
    from pathlib import Path
    import sys
    parser = argparse.ArgumentParser(description="Import all .xlsx files from the data directory")
    default_dir = Path(__file__).resolve().parents[2] / "data"
    parser.add_argument("--dir", default=str(default_dir), help="Directory containing .xlsx files")
    parser.add_argument("--created-by", type=int, default=None, help="User ID for created_by metadata")
    args = parser.parse_args()
    data_dir = Path(args.dir)
    if not data_dir.exists() or not data_dir.is_dir():
        print(f"❌ Directory not found or not a directory: {data_dir}")
        sys.exit(1)
    files = sorted([p for p in data_dir.iterdir() if p.is_file() and p.suffix.lower() == ".xlsx"])
    if not files:
        print(f"⚠️ No .xlsx files found in {data_dir}")
        return
    success = 0
    failed = 0
    failed_files: list[str] = []
    for fp in files:
        print(f"➡️ Processing: {fp.name}")
        try:
            import_excel(str(fp), created_by_user_id=args.created_by)
            success += 1
        except Exception as e:
            print(f"❌ Error importing {fp.name}: {e}")
            failed += 1
            failed_files.append(fp.name)
            continue
    total = len(files)
    print(f"✅ Finished. Total: {total}, Succeeded: {success}, Failed: {failed}")
    if failed_files:
        print("Failed files:")
        for name in failed_files:
            print(f"- {name}")


if __name__ == "__main__":
    _main()
