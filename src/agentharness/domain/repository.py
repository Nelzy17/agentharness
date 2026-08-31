"""The only data access path in the codebase. Tools call it; nothing else does.

Fixtures are read from disk once at import. Swapping JSON for a real database is
a change to this file and no other.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from agentharness.domain.models import (
    Document,
    Followup,
    Meeting,
    Physician,
    Product,
    WriteAttribution,
)

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
DOCS_DIR = DATA_DIR / "docs"

_HONORIFICS = {"dr", "doctor", "prof", "professor"}


@dataclass(frozen=True)
class Resolution[T]:
    """A user-supplied name resolved against the store.

    Exactly one of three outcomes: a single match, nothing, or several. The
    caller is expected to branch on all three, which is why the ambiguous case
    carries the candidate names rather than silently picking the first.
    """

    match: T | None = None
    candidates: tuple[str, ...] = ()

    @property
    def found(self) -> bool:
        return self.match is not None

    @property
    def ambiguous(self) -> bool:
        return len(self.candidates) > 1


_physicians: list[Physician] = []
_products: list[Product] = []
_meetings: list[Meeting] = []
_followups: list[Followup] = []
_documents: list[Document] = []


def _read_json(filename: str) -> list[dict]:
    return json.loads((DATA_DIR / filename).read_text(encoding="utf-8"))


def _parse_document(path: Path) -> Document:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"{path.name} is missing front matter")
    end = lines.index("---", 1)
    front_matter = dict(
        (key.strip(), value.strip())
        for key, _, value in (line.partition(":") for line in lines[1:end])
    )
    return Document(
        doc_id=path.stem,
        product=front_matter["product"],
        title=front_matter["title"],
        body="\n".join(lines[end + 1 :]).strip(),
    )


def reset() -> None:
    """Reload the fixtures from disk, discarding anything written in-process."""
    global _physicians, _products, _meetings, _followups, _documents
    _physicians = [Physician(**record) for record in _read_json("physicians.json")]
    _products = [Product(**record) for record in _read_json("products.json")]
    _meetings = [Meeting(**record) for record in _read_json("meetings.json")]
    _followups = [Followup(**record) for record in _read_json("followups.json")]
    _documents = sorted(
        (_parse_document(path) for path in DOCS_DIR.glob("*.md")),
        key=lambda document: document.doc_id,
    )


def _tokens(text: str) -> set[str]:
    """Name parts of `text`, honorifics dropped and hyphenated parts split out.

    Splitting hyphens is what makes "Chen" match Daniel Chen-Ruiz as well as
    Evelyn Chen, so the ambiguity is reported rather than missed.
    """
    tokens: set[str] = set()
    for word in re.sub(r"[^a-z0-9\s-]", " ", text.lower()).split():
        word = word.strip("-")
        if not word or word in _HONORIFICS:
            continue
        tokens.add(word)
        if "-" in word:
            tokens.update(part for part in word.split("-") if part)
    return tokens


def _resolve[T](
    query: str, records: Sequence[T], name_of: Callable[[T], str]
) -> Resolution[T]:
    wanted = _tokens(query)
    matches = [record for record in records if wanted <= _tokens(name_of(record))]
    if len(matches) == 1:
        return Resolution(match=matches[0])
    # Nothing matched, or several did. A query that normalises to no tokens at
    # all ("Dr.") is a subset of every name and so lands here as ambiguous,
    # which is the honest answer to a request that named nobody.
    return Resolution(candidates=tuple(name_of(record) for record in matches))


def resolve_physician(name: str) -> Resolution[Physician]:
    return _resolve(name, _physicians, lambda physician: physician.full_name)


def resolve_product(name: str) -> Resolution[Product]:
    return _resolve(name, _products, lambda product: product.name)


def meetings_for(physician_id: str, limit: int) -> list[Meeting]:
    """Meetings for one physician, most recent first."""
    matches = [meeting for meeting in _meetings if meeting.physician_id == physician_id]
    matches.sort(key=lambda meeting: meeting.date, reverse=True)
    return matches[:limit]


def open_followups_for(physician_id: str) -> list[Followup]:
    return [
        followup
        for followup in _followups
        if followup.physician_id == physician_id and followup.status == "open"
    ]


def documents(product_name: str | None = None) -> list[Document]:
    if product_name is None:
        return list(_documents)
    return [document for document in _documents if document.product == product_name]


def append_followup(
    physician_id: str,
    description: str,
    due_date: str,
    attribution: WriteAttribution,
) -> Followup:
    """Append a follow-up to the in-memory store and return it.

    Attribution is required rather than optional. Writes execute without a human
    saying yes, so the answer to "what stops this putting nonsense in a
    physician's record" is bounded blast radius plus the ability to undo it --
    and undoing requires knowing which run did what. A signature that allowed an
    unattributed write would leave a record nothing could reverse.
    """
    next_number = 1 + max(
        int(followup.followup_id.rsplit("-", 1)[1]) for followup in _followups
    )
    followup = Followup(
        followup_id=f"fu-{next_number:03d}",
        physician_id=physician_id,
        description=description,
        due_date=due_date,
        status="open",
        created_by_run_id=attribution.run_id,
        created_by_tool_call_id=attribution.tool_call_id,
    )
    _followups.append(followup)
    return followup


def revoke_run(run_id: str) -> list[str]:
    """Remove every follow-up created by one run. Returns the ids removed.

    This is the mechanism that makes "reversible" true rather than aspirational,
    and it is the only removal in this module. It is scoped by construction: a
    run_id that matches nothing removes nothing, and fixture records carry no
    attribution at all, so they cannot be reached by any run_id whatsoever.

    The empty run_id is refused. Falsy input reaching a filter is the classic
    way a scoped delete becomes an unscoped one, and this is the wrong function
    to discover that in.
    """
    global _followups
    if not run_id:
        raise ValueError("revoke_run needs a run_id; refusing to match on nothing")
    removed = [
        followup.followup_id
        for followup in _followups
        if followup.created_by_run_id == run_id
    ]
    _followups = [
        followup for followup in _followups if followup.created_by_run_id != run_id
    ]
    return removed


def all_followups() -> list[Followup]:
    return list(_followups)


def all_physicians() -> list[Physician]:
    return list(_physicians)


def all_products() -> list[Product]:
    return list(_products)


reset()
