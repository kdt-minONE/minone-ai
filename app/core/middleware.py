from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
import time
import logging
from typing import Any

logger = logging.getLogger(__name__)


def setup_middleware(app: FastAPI) -> None:
    # CORS 미들웨어
    app.add_middleware(
        CORSMiddleware,  # type: ignore
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 로깅 미들웨어
    @app.middleware("http")
    async def log_requests(request: Request, call_next: Any) -> Any:
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time

        logger.info(
            f"{request.method} {request.url} "
            f"completed in {process_time:.4f}s "
            f"with status {response.status_code}"
        )

        response.headers["X-Process-Time"] = str(process_time)
        return response
