"""Domain entities and the result shapes the tool functions return.

Dates are plain strings. The entities describe fixture records, not arguments,
and format validation of a model-supplied date belongs to the per-tool argument
models in M1 rather than here.
"""

from typing import Literal

from pydantic import BaseModel, Field


class Physician(BaseModel):
    physician_id: str
    full_name: str
    last_name: str
    specialty: str
    institution: str
    preferred_contact: str
    notes: str


class Product(BaseModel):
    product_id: str
    name: str
    indication: str
    status: str


class Meeting(BaseModel):
    meeting_id: str
    physician_id: str
    date: str
    product_discussed: str
    summary: str
    open_questions: list[str]


class Followup(BaseModel):
    followup_id: str
    physician_id: str
    description: str
    due_date: str
    status: Literal["open", "closed"]
    # Populated by the write tool from M6. Present now so the schema does not
    # churn once writes are attributed.
    created_by_run_id: str | None = None
    created_by_tool_call_id: str | None = None


class Document(BaseModel):
    doc_id: str
    product: str
    title: str
    body: str


class DocSnippet(BaseModel):
    doc_id: str
    product: str
    title: str
    snippet: str
    score: int


# Every tool result carries a status so the caller can distinguish outcomes
# structurally rather than by inspecting whether a list came back empty.
# "empty" means the subject resolved but has no records; it is not a failure.

class PhysicianProfileResult(BaseModel):
    status: Literal["ok", "not_found", "ambiguous"]
    message: str
    physician: Physician | None = None
    candidates: list[str] = Field(default_factory=list)


class MeetingsResult(BaseModel):
    status: Literal["ok", "empty", "not_found", "ambiguous"]
    message: str
    physician_name: str | None = None
    meetings: list[Meeting] = Field(default_factory=list)
    candidates: list[str] = Field(default_factory=list)


class FollowupsResult(BaseModel):
    status: Literal["ok", "empty", "not_found", "ambiguous"]
    message: str
    physician_name: str | None = None
    followups: list[Followup] = Field(default_factory=list)
    candidates: list[str] = Field(default_factory=list)


class DocSearchResult(BaseModel):
    status: Literal["ok", "empty", "not_found", "ambiguous"]
    message: str
    results: list[DocSnippet] = Field(default_factory=list)
    candidates: list[str] = Field(default_factory=list)


class CreateFollowupResult(BaseModel):
    status: Literal["ok", "not_found", "ambiguous"]
    message: str
    followup: Followup | None = None
    candidates: list[str] = Field(default_factory=list)
