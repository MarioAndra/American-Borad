"""Regression checks for standalone Excel importer cognitive-level mapping.

Run:  python -m app.scripts.test_excel_import_compat

Verifies that app/scripts/import_questions_from_excel.py delegates to the
shared resolver (cognitive_level_from_value) instead of its own hardcoded
COGNITIVE_MAP, and that canonical, legacy, and spreadsheet tokens all resolve
consistently. Pure tests — no model, no database.
"""
from __future__ import annotations

import sys

from app.models.enums import cognitive_level_from_value

_RESULTS: list[tuple[str, bool, str]] = []


def _ok(name: str) -> None:
    _RESULTS.append((name, True, ""))


def _fail(name: str, msg: str) -> None:
    _RESULTS.append((name, False, msg))


def _hasattr(module, name: str) -> bool:
    return hasattr(module, name)


def test_no_hardcoded_mapping() -> None:
    name = "1. standalone importer has no COGNITIVE_MAP"
    try:
        from app.scripts import import_questions_from_excel as mod

        assert not _hasattr(mod, "COGNITIVE_MAP"), "COGNITIVE_MAP still defined"
        _ok(name)
    except Exception as exc:
        _fail(name, str(exc))


def test_uses_shared_resolver() -> None:
    name = "2. standalone importer uses cognitive_level_from_value"
    try:
        from app.scripts import import_questions_from_excel as mod

        assert _hasattr(mod, "cognitive_level_from_value")
        assert mod.cognitive_level_from_value is cognitive_level_from_value
        _ok(name)
    except Exception as exc:
        _fail(name, str(exc))


def test_resolver_accepts_all_tokens() -> None:
    name = "3. resolver accepts canonical, legacy, and spreadsheet tokens"
    try:
        expected = {
            "RememberUnderstand": "RememberUnderstand",
            "Apply": "Apply",
            "Analyze": "Analyze",
            "Evaluate": "Evaluate",
            "Create": "Create",
            "Knowledge": "RememberUnderstand",
            "Application": "Apply",
            "Analysis": "Analyze",
            "Remember": "RememberUnderstand",
            "Understand": "RememberUnderstand",
        }
        for token, canonical in expected.items():
            assert cognitive_level_from_value(token).value == canonical, token
        for token in (" remember ", "APPLY", "knowledge"):
            assert cognitive_level_from_value(token).value is not None, token
        _ok(name)
    except Exception as exc:
        _fail(name, str(exc))


def test_rejects_unknown_tokens() -> None:
    name = "4. unknown cognitive-level tokens rejected"
    try:
        for token in ("Foo", "Bloom", ""):
            try:
                cognitive_level_from_value(token)
                raise AssertionError(f"{token!r} was accepted")
            except ValueError:
                pass
        _ok(name)
    except Exception as exc:
        _fail(name, str(exc))


_ALL_TESTS = [
    test_no_hardcoded_mapping,
    test_uses_shared_resolver,
    test_resolver_accepts_all_tokens,
    test_rejects_unknown_tokens,
]


def main() -> int:
    for fn in _ALL_TESTS:
        fn()

    passed = sum(1 for _, ok, _ in _RESULTS if ok)
    failed = sum(1 for _, ok, _ in _RESULTS if not ok)

    print()
    for name, ok, msg in _RESULTS:
        tag = "PASS" if ok else "FAIL"
        line = f"  [{tag}] {name}"
        if msg:
            line += f"  --  {msg}"
        print(line)

    print()
    print("-" * 60)
    print(f"  {passed} passed, {failed} failed, {len(_RESULTS)} total")
    print("-" * 60)

    if failed:
        print("\nSOME TESTS FAILED")
        return 1

    print("\nALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
