import logging
from typing import Any, Dict
from rest_framework.views import exception_handler
from rest_framework.exceptions import APIException, ValidationError, NotAuthenticated, PermissionDenied
from rest_framework import status

logger = logging.getLogger(__name__)


class AppError(APIException):
    status_code = status.HTTP_400_BAD_REQUEST
    default_detail = "An error occurred."
    default_code = "app_error"


class ProcessingError(AppError):
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    default_detail = "Failed to process the request."
    default_code = "processing_error"


class ResourceNotFound(AppError):
    status_code = status.HTTP_404_NOT_FOUND
    default_detail = "Requested resource not found."
    default_code = "not_found"


def drf_exception_handler(exc: Exception, context: Dict[str, Any]):
    """Global DRF exception handler returning consistent error payloads."""
    response = exception_handler(exc, context)

    if response is not None:
        # Normalize known DRF exceptions
        detail = response.data
        if isinstance(exc, ValidationError):
            message = "Validation failed"
            code = "validation_error"
        elif isinstance(exc, NotAuthenticated):
            message = "Authentication credentials were not provided or are invalid"
            code = "not_authenticated"
        elif isinstance(exc, PermissionDenied):
            message = "You do not have permission to perform this action"
            code = "permission_denied"
        else:
            message = detail.get("detail", "An error occurred") if isinstance(detail, dict) else "An error occurred"
            code = getattr(exc, 'default_code', 'error')

        response.data = {
            "success": False,
            "error": {
                "code": code,
                "message": message,
                "details": detail,
            }
        }
        return response

    # Unhandled exceptions -> 500
    logger.exception("Unhandled exception in API", exc_info=exc)
    from rest_framework.response import Response
    return Response({
        "success": False,
        "error": {
            "code": "server_error",
            "message": "Internal server error",
        }
    }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
