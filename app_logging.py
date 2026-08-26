import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
LOG_DIR = APP_DIR / "data" / "logs"
LOG_FILE = LOG_DIR / "app.log"

_LOGGER_NAME = "ai_learning_platform"
_configured = False


def _resolve_level() -> int:
    """환경변수 LOG_LEVEL로 로그 레벨을 조정할 수 있게 합니다(기본 INFO)."""
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    return getattr(logging, level_name, logging.INFO)


def get_logger() -> logging.Logger:
    """앱 전역에서 공유하는 로거를 반환합니다(최초 호출 시 1회 설정)."""
    global _configured

    logger = logging.getLogger(_LOGGER_NAME)

    if _configured:
        return logger

    logger.setLevel(_resolve_level())
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=1_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        pass

    _configured = True
    return logger


def log_exception(message: str, error: BaseException) -> None:
    """예외를 스택 트레이스와 함께 로그로 남기는 단축 함수."""
    logger = get_logger()
    logger.error("%s: %s", message, error, exc_info=True)


def log_event(message: str, level: str = "info") -> None:
    """일반 이벤트(로그인, 업로드, AI 호출 등)를 기록하는 단축 함수."""
    logger = get_logger()
    log_method = getattr(logger, level.lower(), logger.info)
    log_method(message)
