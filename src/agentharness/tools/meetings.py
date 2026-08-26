from agentharness.domain import repository
from agentharness.domain.models import MeetingsResult
from agentharness.tools._resolution import ambiguous_message, not_found_message


def get_previous_meetings(physician_name: str, limit: int = 5) -> MeetingsResult:
    resolution = repository.resolve_physician(physician_name)
    if resolution.ambiguous:
        return MeetingsResult(
            status="ambiguous",
            message=ambiguous_message("physician", physician_name, resolution.candidates),
            candidates=list(resolution.candidates),
        )
    if not resolution.found:
        return MeetingsResult(
            status="not_found",
            message=not_found_message("physician", physician_name),
        )
    physician = resolution.match
    meetings = repository.meetings_for(physician.physician_id, limit)
    if not meetings:
        return MeetingsResult(
            status="empty",
            message=f"No meetings are on record for {physician.full_name}.",
            physician_name=physician.full_name,
        )
    return MeetingsResult(
        status="ok",
        message=f"{len(meetings)} meeting(s) for {physician.full_name}, most recent first.",
        physician_name=physician.full_name,
        meetings=meetings,
    )
