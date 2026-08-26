"""Substring search over the checked-in product documents.

Deliberately not retrieval. This tool exists to be the untrusted free-text
surface and the zero-results path; it is not to be improved.
"""

import re

from agentharness.domain import repository
from agentharness.domain.models import DocSearchResult, DocSnippet
from agentharness.tools._resolution import ambiguous_message, not_found_message

# Terms shorter than this are ignored, so that a natural-language query is
# scored on "nexovar" and "dosing" rather than on how often the longest
# document happens to say "the".
MIN_TERM_LENGTH = 4
SNIPPET_RADIUS = 200
MAX_RESULTS = 3


def search_product_docs(query: str, product_name: str | None = None) -> DocSearchResult:
    documents = repository.documents()
    if product_name is not None:
        resolution = repository.resolve_product(product_name)
        if resolution.ambiguous:
            return DocSearchResult(
                status="ambiguous",
                message=ambiguous_message("product", product_name, resolution.candidates),
                candidates=list(resolution.candidates),
            )
        if not resolution.found:
            return DocSearchResult(
                status="not_found",
                message=not_found_message("product", product_name),
            )
        documents = repository.documents(resolution.match.name)

    terms = [t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) >= MIN_TERM_LENGTH]
    scored = []
    for document in documents:
        body = document.body.lower()
        heading = f"{document.title} {document.product}".lower()
        score = sum(body.count(term) + 3 * heading.count(term) for term in terms)
        if score:
            scored.append((score, document))
    scored.sort(key=lambda row: (-row[0], row[1].doc_id))

    if not scored:
        return DocSearchResult(
            status="empty",
            message=f"No documents matched '{query}'.",
            candidates=[],
        )
    results = [
        DocSnippet(
            doc_id=document.doc_id,
            product=document.product,
            title=document.title,
            snippet=_snippet(document.body, terms),
            score=score,
        )
        for score, document in scored[:MAX_RESULTS]
    ]
    return DocSearchResult(
        status="ok",
        message=f"{len(results)} document(s) matched '{query}'.",
        results=results,
    )


def _snippet(body: str, terms: list[str]) -> str:
    """A window around the first matching term, whitespace collapsed."""
    lowered = body.lower()
    hits = [position for position in (lowered.find(term) for term in terms) if position >= 0]
    first = min(hits) if hits else 0
    start = max(0, first - SNIPPET_RADIUS)
    end = min(len(body), first + SNIPPET_RADIUS)
    window = " ".join(body[start:end].split())
    return f"{'...' if start else ''}{window}{'...' if end < len(body) else ''}"
