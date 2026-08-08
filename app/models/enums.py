import enum


class DifficultyLevel(str, enum.Enum):
    Easy = "Easy"
    Medium = "Medium"
    Hard = "Hard"


class ExamStatus(str, enum.Enum):
    Pending = "Pending"
    InProgress = "InProgress"
    Completed = "Completed"


class UserRole(str, enum.Enum):
    Student = "Student"
    Admin = "Admin"


class CognitiveLevel(str, enum.Enum):
    RememberUnderstand = "RememberUnderstand"
    Apply = "Apply"
    Analyze = "Analyze"
    Evaluate = "Evaluate"
    Create = "Create"

    # Legacy aliases — resolve to the new taxonomy so untouched call sites
    # (Excel import scripts, adaptive exam service) keep working.
    Knowledge = "RememberUnderstand"
    Application = "Apply"
    Analysis = "Analyze"


# Legacy/alternate spellings accepted by data imports. Keys are lowercased
# tokens; values are canonical members. Remember and Understand were separate
# columns in legacy banks but map to the combined lower level of the model.
COGNITIVE_LEVEL_ALIASES: dict[str, CognitiveLevel] = {
    "remember": CognitiveLevel.RememberUnderstand,
    "understand": CognitiveLevel.RememberUnderstand,
    "knowledge": CognitiveLevel.RememberUnderstand,
    "application": CognitiveLevel.Apply,
    "analysis": CognitiveLevel.Analyze,
}


def cognitive_level_from_value(value: str) -> CognitiveLevel:
    """Resolve a canonical or legacy cognitive-level spelling.

    Accepts the canonical values (RememberUnderstand, Apply, Analyze,
    Evaluate, Create), the pre-Bloom legacy values (Knowledge, Application,
    Analysis), and the tokens used in legacy question-bank spreadsheets
    (Remember, Understand). Raises ``ValueError`` for anything else.
    """
    token = value.strip().lower()
    for member in CognitiveLevel:
        if token == member.value.lower():
            return member
    level = COGNITIVE_LEVEL_ALIASES.get(token)
    if level is not None:
        return level
    raise ValueError(f"Invalid cognitive_level: {value}")


class QuestionType(str, enum.Enum):
    SingleChoice = "SingleChoice"
    MultipleSelect = "MultipleSelect"


class GeneratedQuestionStatus(str, enum.Enum):
    draft = "draft"
    approved = "approved"
    rejected = "rejected"
    auto_approved = "auto_approved"
