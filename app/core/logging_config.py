"""
로깅 설정 모듈
콘솔과 파일에 동시에 로그를 기록하는 설정
"""
import logging
import logging.config
from pathlib import Path


def setup_logging():
    """로깅 설정을 초기화합니다."""

    # 로그 디렉토리 생성
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # 로깅 설정 딕셔너리
    LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "[{asctime}] {levelname:<8} {name}: {message}",
                "style": "{",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "detailed": {
                "format": "[{asctime}] {levelname:<8} {name} [{filename}:{lineno}] {funcName}(): {message}",
                "style": "{",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "simple": {
                "format": "{levelname}: {message}",
                "style": "{"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "DEBUG",
                "formatter": "default",
                "stream": "ext://sys.stdout"
            },
            "file_debug": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "DEBUG",
                "formatter": "detailed",
                "filename": "logs/debug.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "encoding": "utf8"
            },
            "file_info": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "INFO",
                "formatter": "default",
                "filename": "logs/app.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "encoding": "utf8"
            },
            "file_error": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "ERROR",
                "formatter": "detailed",
                "filename": "logs/error.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 10,
                "encoding": "utf8"
            }
        },
        "loggers": {
            # 앱 전체 로거
            "app": {
                "level": "DEBUG",
                "handlers": ["console", "file_debug", "file_info", "file_error"],
                "propagate": False
            },
            # Repository 로거
            "app.repositories": {
                "level": "DEBUG",
                "handlers": ["console", "file_debug", "file_info", "file_error"],
                "propagate": False
            },
            # Service 로거
            "app.services": {
                "level": "DEBUG",
                "handlers": ["console", "file_debug", "file_info", "file_error"],
                "propagate": False
            },
            # API 로거
            "app.api": {
                "level": "INFO",
                "handlers": ["console", "file_info", "file_error"],
                "propagate": False
            },
            # 외부 라이브러리 로거
            "sqlalchemy.engine": {
                "level": "WARNING",
                "handlers": ["file_debug"],
                "propagate": False
            },
            "uvicorn": {
                "level": "INFO",
                "handlers": ["console", "file_info"],
                "propagate": False
            },
            "uvicorn.access": {
                "level": "INFO",
                "handlers": ["file_info"],
                "propagate": False
            }
        },
        "root": {
            "level": "INFO",
            "handlers": ["console", "file_info", "file_error"]
        }
    }

    # 로깅 설정 적용
    logging.config.dictConfig(LOGGING_CONFIG)

    # 앱 시작 로그
    logger = logging.getLogger("app")
    logger.info("로깅 시스템이 초기화되었습니다.")
    logger.info(f"로그 파일 위치: {log_dir.absolute()}")


def get_logger(name: str = None) -> logging.Logger:
    """
    로거 인스턴스를 반환합니다.

    Args:
        name: 로거 이름 (None이면 'app' 사용)

    Returns:
        logging.Logger: 로거 인스턴스
    """
    if name is None:
        name = "app"
    return logging.getLogger(name)
