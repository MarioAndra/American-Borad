from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import AnyUrl, Field
from typing import Optional


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra="ignore")

    APP_NAME: str = "American Board of AI Exam System"
    APP_VERSION: str = "0.1.0"

    # Database URL for Alembic and runtime
    DATABASE_URL: Optional[AnyUrl] = None

    # JWT
    JWT_SECRET_KEY: str = Field(default="change-me", description="Secret key for JWT signing")
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 30

    # Security policies
    EMAIL_VERIFICATION_TOKEN_EXPIRE_MINUTES: int = 30
    EMAIL_VERIFICATION_CODE_EXPIRE_MINUTES: int = 10
    EMAIL_VERIFICATION_MAX_ATTEMPTS: int = 5
    RESEND_VERIFICATION_RATE_LIMIT_MINUTES: int = 5
    RESEND_VERIFICATION_MAX_PER_HOUR: int = 5
    OTP_CODE_LENGTH: int = 6
    FAILED_LOGIN_MAX_ATTEMPTS: int = 5
    ACCOUNT_LOCKOUT_MINUTES: int = 15
    PASSWORD_RESET_TOKEN_EXPIRE_MINUTES: int = 30

    # App URLs
    BASE_URL: str = "http://localhost:8000"

    # Phase II Adaptive Exam
    PHASE2_ENABLED: bool = True
    PHASE2_PHASE_ID: int = 2
    PHASE2_MAX_QUESTIONS: int = 20
    PHASE2_PASSING_SCORE: float = 75.0
    PHASE2_INITIAL_THETA: float = 0.0

    # Mail
    MAIL_HOST: str = "smtp.gmail.com"
    MAIL_PORT: int = 587
    MAIL_USERNAME: Optional[str] = None
    MAIL_PASSWORD: Optional[str] = None
    MAIL_FROM_NAME: str = "American Board System"
    MAIL_FROM: Optional[str] = None
    MAIL_USE_TLS: bool = Field(default=True, env=["MAIL_USE_TLS", "MAIL_TLS"])
    MAIL_ENCRYPTION: Optional[str] = None

    # RAG — Ingestion & Retrieval
    RAG_ENABLED: bool = True
    RAG_SOURCE_ROOT: str = "./courses"
    RAG_EMBEDDING_PROVIDER: str = "openai"
    RAG_EMBEDDING_MODEL: str = "text-embedding-3-small"
    RAG_GENERATION_PROVIDER: str = "openai"
    RAG_GENERATION_MODEL: str = "gpt-4o-mini"
    RAG_CHUNK_SIZE: int = 1200
    RAG_CHUNK_OVERLAP: int = 180
    RAG_RETRIEVAL_TOP_K: int = 8
    RAG_MIN_SIMILARITY: float = 0.2
    RAG_ALLOWED_SOURCE_TYPES: str = "page,resource,pdf,transcript"

    # Phase II — Topic/SubTopic Adaptive Controls
    PHASE2_SUBTOPIC_BASE_QUESTION_COUNT: int = 4
    PHASE2_SUBTOPIC_GENERATED_QUESTION_COUNT: int = 1
    PHASE2_RESET_THETA_PER_TOPIC: bool = True
    PHASE2_GENERATED_THETA_STEP_FACTOR: float = 0.5

    # Generated MCQ Constraints
    GENERATED_MCQ_OPTION_COUNT: int = 4
    GENERATED_MCQ_REQUIRE_CITATIONS: bool = True
    GENERATED_MCQ_REQUIRE_DISTRACTOR_RATIONALE: bool = True
    RAG_REVIEW_REQUIRED: bool = False
    RAG_ALLOW_VECTOR_FALLBACK: bool = False
    RAG_OCR_ENABLED: bool = False
    RAG_LANGGRAPH_ENABLED: bool = False

    # External API keys
    OPENAI_API_KEY: Optional[str] = None

    # Weaviate vector store
    WEAVIATE_URL: str = "http://localhost:8080"

    # Gemini (alternative provider for RAG)
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_EMBEDDING_MODEL: str = "text-embedding-002"
    GEMINI_GENERATION_MODEL: str = "gemini-2.5-flash"

    # Groq (OpenAI-compatible provider for RAG)
    GROQ_API_KEY: Optional[str] = None
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    GROQ_EMBEDDING_MODEL: str = "mxbai-embed-large"
    GROQ_GENERATION_MODEL: str = "llama-3.3-70b-versatile"

    # Logging
    LOG_LEVEL: str = "INFO"


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()  # type: ignore[call-arg]
    return _settings
