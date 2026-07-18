from __future__ import annotations

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse


class AppException(Exception):
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    detail: str = "Internal server error"

    def __init__(self, detail: str | None = None) -> None:
        if detail:
            self.detail = detail


class NotFoundException(AppException):
    status_code = status.HTTP_404_NOT_FOUND
    detail = "Resource not found"


class ValidationException(AppException):
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    detail = "Validation failed"


class ConflictException(AppException):
    status_code = status.HTTP_409_CONFLICT
    detail = "Resource already exists"


class UnauthorizedException(AppException):
    status_code = status.HTTP_401_UNAUTHORIZED
    detail = "Not authenticated"


class ForbiddenException(AppException):
    status_code = status.HTTP_403_FORBIDDEN
    detail = "Insufficient permissions"


_EXCEPTION_HANDLERS: dict[type[AppException], int] = {
    NotFoundException: status.HTTP_404_NOT_FOUND,
    ValidationException: status.HTTP_422_UNPROCESSABLE_ENTITY,
    ConflictException: status.HTTP_409_CONFLICT,
    UnauthorizedException: status.HTTP_401_UNAUTHORIZED,
    ForbiddenException: status.HTTP_403_FORBIDDEN,
}


def register_exception_handlers(app: FastAPI) -> None:
    for exc_cls, http_code in _EXCEPTION_HANDLERS.items():

        def _handler(request, exc: AppException, code: int = http_code) -> JSONResponse:
            return JSONResponse(status_code=code, content={"detail": exc.detail or exc_cls.detail})

        app.add_exception_handler(exc_cls, _handler)

    @app.exception_handler(AppException)
    async def _app_exception_handler(request, exc: AppException) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
