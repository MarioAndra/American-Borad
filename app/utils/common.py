from __future__ import annotations

import re
from datetime import datetime, timezone


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def slugify(text: str, max_len: int = 120) -> str:
    s = text.lower().strip()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[-\s]+", "-", s)
    return s[:max_len].rstrip("-")
