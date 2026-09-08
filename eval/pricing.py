"""Token prices, with the date they were checked.

A cost table without a date is a number that silently rots. These move fast:
Luna's input price dropped roughly 80% in a single day in July 2026, and Sol's
rate below is promotional through at least 2026-11-21. Any cost figure computed
here is only as current as PRICES_CHECKED_ON, and the reports print that date
beside the total so a stale number cannot be mistaken for a live one.
"""

from dataclasses import dataclass

PRICES_CHECKED_ON = "2026-08-31"
PROMOTIONAL_UNTIL = {"gpt-5.6-sol": "2026-11-21"}


@dataclass(frozen=True)
class Pricing:
    """US dollars per million tokens."""

    input_per_million: float
    cached_input_per_million: float
    output_per_million: float


PRICING: dict[str, Pricing] = {
    "gpt-5.6-luna": Pricing(
        input_per_million=0.20,
        cached_input_per_million=0.02,
        output_per_million=1.20,
    ),
    "gpt-5.6-sol": Pricing(
        input_per_million=5.00,
        cached_input_per_million=0.50,
        output_per_million=30.00,
    ),
}


def cost_of(
    model: str,
    prompt_tokens: int,
    cached_tokens: int,
    completion_tokens: int,
) -> float | None:
    """Cost in dollars, or None for a model with no recorded price.

    `cached_tokens` is a subset of `prompt_tokens`, not an addition to it, so
    the uncached remainder is what gets charged at the standard input rate.

    Cache *writes* are treated as ordinary input here. If the provider charges a
    premium for them -- some do -- this underestimates by that premium on the
    first call of each run, and the reports say so rather than quietly rounding
    in our favour.
    """
    pricing = PRICING.get(model)
    if pricing is None:
        return None
    uncached = max(0, prompt_tokens - cached_tokens)
    return (
        uncached * pricing.input_per_million
        + cached_tokens * pricing.cached_input_per_million
        + completion_tokens * pricing.output_per_million
    ) / 1_000_000


def price_note(model: str) -> str:
    note = f"prices checked {PRICES_CHECKED_ON}"
    if model in PROMOTIONAL_UNTIL:
        note += f"; {model} rate is promotional through {PROMOTIONAL_UNTIL[model]}"
    if model not in PRICING:
        note += f"; no price recorded for {model}, cost omitted"
    return note
