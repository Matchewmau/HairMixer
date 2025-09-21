from typing import Any, Dict, Optional


def track_event_safe(
    analytics_service,
    user: Any,
    event_type: str,
    event_data: Optional[Dict[str, Any]] = None,
    request: Any = None,
) -> None:
    if not analytics_service:
        return
    try:
        analytics_service.track_event(
            user=user,
            event_type=event_type,
            event_data=event_data or {},
            request=request,
        )
    except Exception:
        # Swallow analytics errors; never impact main request flow
        return
