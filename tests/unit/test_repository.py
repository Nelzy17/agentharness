"""Repository-level tests: fixture integrity and name resolution."""

import pytest

from agentharness.domain import repository
from agentharness.domain.models import WriteAttribution

ATTRIBUTION = WriteAttribution(run_id="test-run", tool_call_id="call_test")


@pytest.fixture(autouse=True)
def fresh_store():
    repository.reset()


def test_fixture_counts_are_what_the_eval_set_expects():
    assert len(repository.all_physicians()) == 6
    assert len(repository.all_products()) == 4
    assert len(repository.documents()) == 8
    assert len(repository.meetings_for("phy-001", 99)) == 3


def test_document_counts_per_product():
    counts = {
        product.name: len(repository.documents(product.name))
        for product in repository.all_products()
    }
    assert counts == {"Nexovar": 4, "Cardizyn": 3, "Zelmarin": 0, "Trivastol": 1}


def test_every_document_names_a_real_product():
    product_names = {product.name for product in repository.all_products()}
    assert product_names.issuperset(
        {document.product for document in repository.documents()}
    )


def test_every_meeting_and_followup_names_a_real_physician():
    physician_ids = {p.physician_id for p in repository.all_physicians()}
    for physician_id in physician_ids:
        assert all(
            meeting.physician_id in physician_ids
            for meeting in repository.meetings_for(physician_id, 99)
        )
        assert all(
            followup.physician_id in physician_ids
            for followup in repository.open_followups_for(physician_id)
        )


@pytest.mark.parametrize(
    "query, expected",
    [
        ("Dr. Evelyn Chen", "phy-001"),
        ("evelyn chen", "phy-001"),
        ("  EVELYN   CHEN  ", "phy-001"),
        ("Patel", "phy-002"),
        ("Dr Patel", "phy-002"),
        ("Doctor Okafor", "phy-004"),
        ("chen-ruiz", "phy-006"),
        ("Ruiz", "phy-006"),
    ],
)
def test_resolution_is_case_and_honorific_insensitive(query, expected):
    assert repository.resolve_physician(query).match.physician_id == expected


def test_hyphenated_last_name_makes_chen_ambiguous():
    resolution = repository.resolve_physician("Chen")
    assert not resolution.found
    assert resolution.ambiguous
    assert resolution.candidates == ("Evelyn Chen", "Daniel Chen-Ruiz")


def test_unknown_name_is_neither_found_nor_ambiguous():
    resolution = repository.resolve_physician("Nakamura")
    assert not resolution.found
    assert not resolution.ambiguous
    assert resolution.candidates == ()


def test_reset_discards_in_process_writes():
    repository.append_followup("phy-003", "temporary", "2026-09-01", ATTRIBUTION)
    assert len(repository.open_followups_for("phy-003")) == 1
    repository.reset()
    assert repository.open_followups_for("phy-003") == []


def test_appended_followup_ids_do_not_collide():
    first = repository.append_followup("phy-003", "one", "2026-09-01", ATTRIBUTION)
    second = repository.append_followup("phy-003", "two", "2026-09-02", ATTRIBUTION)
    assert (first.followup_id, second.followup_id) == ("fu-006", "fu-007")
