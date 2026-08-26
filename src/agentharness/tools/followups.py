from agentharness.domain import repository
from agentharness.domain.models import CreateFollowupResult, FollowupsResult
from agentharness.tools._resolution import ambiguous_message, not_found_message


def get_open_followups(physician_name: str) -> FollowupsResult:
    resolution = repository.resolve_physician(physician_name)
    if resolution.ambiguous:
        return FollowupsResult(
            status="ambiguous",
            message=ambiguous_message("physician", physician_name, resolution.candidates),
            candidates=list(resolution.candidates),
        )
    if not resolution.found:
        return FollowupsResult(
            status="not_found",
            message=not_found_message("physician", physician_name),
        )
    physician = resolution.match
    followups = repository.open_followups_for(physician.physician_id)
    if not followups:
        return FollowupsResult(
            status="empty",
            message=f"No open follow-ups are on record for {physician.full_name}.",
            physician_name=physician.full_name,
        )
    return FollowupsResult(
        status="ok",
        message=f"{len(followups)} open follow-up(s) for {physician.full_name}.",
        physician_name=physician.full_name,
        followups=followups,
    )


def create_followup(
    physician_name: str, description: str, due_date: str
) -> CreateFollowupResult:
    resolution = repository.resolve_physician(physician_name)
    if resolution.ambiguous:
        return CreateFollowupResult(
            status="ambiguous",
            message=ambiguous_message("physician", physician_name, resolution.candidates),
            candidates=list(resolution.candidates),
        )
    if not resolution.found:
        return CreateFollowupResult(
            status="not_found",
            message=not_found_message("physician", physician_name),
        )
    physician = resolution.match
    followup = repository.append_followup(physician.physician_id, description, due_date)
    return CreateFollowupResult(
        status="ok",
        message=f"Created follow-up {followup.followup_id} for {physician.full_name}, due {followup.due_date}.",
        followup=followup,
    )
