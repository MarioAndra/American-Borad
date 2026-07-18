"""Seed admin user into the database.

Usage:
    python -m app.scripts.seed_admin --email admin@example.com --password admin123
    python -m app.scripts.seed_admin --email admin@example.com --password admin123 --name "Super Admin"
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

from app.core.security import get_password_hash
from app.db.session import SessionLocal
from app.models import User, UserRole


def seed_admin(email: str, password: str, full_name: str = "Admin") -> User:
    db = SessionLocal()
    try:
        existing = db.query(User).filter(User.email == email).first()
        if existing:
            print(f"Admin user already exists: {email} (id={existing.id}, role={existing.role.value})")
            return existing

        user = User(
            full_name=full_name,
            email=email,
            hashed_password=get_password_hash(password),
            role=UserRole.Admin,
            is_active=True,
            is_verified=True,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        print(f"Admin user created: {email} (id={user.id})")
        return user
    finally:
        db.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seed an admin user into the database")
    parser.add_argument("--email", required=True, help="Admin email address")
    parser.add_argument("--password", required=True, help="Admin password")
    parser.add_argument("--name", default="Admin", help="Admin full name (default: Admin)")
    args = parser.parse_args(argv)

    try:
        seed_admin(args.email, args.password, args.name)
        return 0
    except Exception as e:
        print(f"Failed to seed admin: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
