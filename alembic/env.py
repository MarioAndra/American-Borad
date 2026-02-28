from __future__ import annotations

from logging.config import fileConfig
import os
import sys
from sqlalchemy import engine_from_config, pool
from alembic import context

# Ensure project root is on sys.path (Docker /app or local project root)
sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

try:
    from app.core.config import get_settings
    settings = get_settings()
    settings_url = str(settings.DATABASE_URL) if settings.DATABASE_URL else None
except Exception:
    settings_url = None

db_url = os.getenv("ALEMBIC_DATABASE_URL") or settings_url or os.getenv("DATABASE_URL")
if db_url:
    config.set_main_option("sqlalchemy.url", db_url)

from app.db.base  import Base  # noqa: E402
import app.models
target_metadata = Base.metadata

def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_schemas=False,
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )
        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()