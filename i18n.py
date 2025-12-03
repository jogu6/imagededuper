"""Simple runtime translation helper that reads locale JSON files."""

from __future__ import annotations

import json
from pathlib import Path

from config import LANGUAGE

_LOCALES_DIR = Path(__file__).resolve().parent / "locales"
_DEFAULT_LANG = "en"


def _load_locale(lang: str) -> dict[str, str]:
    try:
        path = _LOCALES_DIR / f"{lang}.json"
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


_BASE_TRANSLATIONS = _load_locale(_DEFAULT_LANG)
_TARGET_TRANSLATIONS = (
    _BASE_TRANSLATIONS if LANGUAGE.lower() == _DEFAULT_LANG else _load_locale(LANGUAGE.lower())
)


def t(key: str, **kwargs) -> str:
    """Return the localized string for the given key."""
    template = _TARGET_TRANSLATIONS.get(key) or _BASE_TRANSLATIONS.get(key) or key
    return template.format(**kwargs) if kwargs else template
