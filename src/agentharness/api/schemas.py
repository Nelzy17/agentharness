"""Request and response models. Shapes only -- no behaviour lives here."""

from typing import Any

from pydantic import BaseModel, Field


class RunRequest(BaseModel):
    goal: str = Field(min_length=1, description="What the representative wants done.")


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    cache_write_tokens: int
    total_tokens: int


class RunResponse(BaseModel):
    run_id: str
    terminal_reason: str
    reason_text: str
    answer: str | None
    sources: list[str]
    insufficient_information: bool
    route: str
    iterations: int
    usage: Usage


class RunDetail(BaseModel):
    """A run and every step it took, as stored."""

    run: dict[str, Any]
    steps: list[dict[str, Any]]
