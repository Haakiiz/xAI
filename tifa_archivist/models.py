from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


Label = Literal[
    "original_game",
    "wallpaper",
    "cosplay",
    "fanart_2d",
    "render_3d",
    "sexy_sfw",
    "nsfw",
    "other",
    "discard",
]


class ClassificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: Label
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)
    is_underage_or_ambiguous: bool
