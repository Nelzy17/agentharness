from agentharness.domain import repository
from agentharness.domain.models import PhysicianProfileResult
from agentharness.tools._resolution import ambiguous_message, not_found_message


def get_physician_profile(physician_name: str) -> PhysicianProfileResult:
    resolution = repository.resolve_physician(physician_name)
    if resolution.ambiguous:
        return PhysicianProfileResult(
            status="ambiguous",
            message=ambiguous_message("physician", physician_name, resolution.candidates),
            candidates=list(resolution.candidates),
        )
    if not resolution.found:
        return PhysicianProfileResult(
            status="not_found",
            message=not_found_message("physician", physician_name),
        )
    return PhysicianProfileResult(
        status="ok",
        message=f"Profile for {resolution.match.full_name}.",
        physician=resolution.match,
    )
