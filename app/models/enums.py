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
    Knowledge = "Knowledge"
    Application = "Application"
    Analysis = "Analysis"


class QuestionType(str, enum.Enum):
    SingleChoice = "SingleChoice"
    MultipleSelect = "MultipleSelect"


class GeneratedQuestionStatus(str, enum.Enum):
    draft = "draft"
    approved = "approved"
    rejected = "rejected"
    auto_approved = "auto_approved"
