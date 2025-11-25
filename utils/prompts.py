"""Prompt utilities for future text conditioning."""
from __future__ import annotations

from urllib.parse import urlparse

import pandas as pd


def build_prompt(row: pd.Series) -> str:
    """Construct a lightweight text prompt from a CSV row."""
    url = str(row.get("PAGE_URL", ""))
    tags = str(row.get("TAGS", ""))

    parsed = urlparse(url)
    slug = parsed.path.strip("/").split("/")[-1]
    title = slug.replace("-", " ") if slug else "minecraft build"

    prompt_parts = [title]
    if tags and tags.lower() != "nan":
        prompt_parts.append(tags)

    return " | ".join(part for part in prompt_parts if part)
