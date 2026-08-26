import hashlib
import json
import os
import time
from pathlib import Path

import streamlit as st

import storage

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
LOGIN_ATTEMPTS_FILE = DATA_DIR / "login_attempts.json"
USAGE_FILE = DATA_DIR / "usage.json"

LOGIN_MAX_ATTEMPTS = 5
LOGIN_LOCKOUT_SECONDS = 15 * 60

DAILY_AI_CALL_LIMIT = 50
MAX_PDF_SIZE_MB = 20


def hash_password(password: str) -> str:
    """PBKDF2-HMAC-SHA256으로 비밀번호를 해싱합니다. 반환 형식: 'salt_hex$hash_hex'"""
    salt = os.urandom(16)
    derived = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        200_000,
    )
    return salt.hex() + "$" + derived.hex()


def verify_password(password: str, stored_hash: str) -> bool:
    """저장된 해시와 입력 비밀번호가 일치하는지 확인합니다."""
    try:
        salt_hex, hash_hex = str(stored_hash).split("$", 1)
        salt = bytes.fromhex(salt_hex)
    except (ValueError, AttributeError):
        return False

    derived = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        200_000,
    )
    return derived.hex() == hash_hex


def _ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path, default: dict) -> dict:
    _ensure_data_dir()
    data = storage.read_json(path, default)
    return data if isinstance(data, dict) else dict(default)


def _save_json(path: Path, data: dict):
    _ensure_data_dir()
    storage.write_json(path, data)


def _load_login_attempts() -> dict:
    return _load_json(LOGIN_ATTEMPTS_FILE, {})


def _save_login_attempts(data: dict):
    _save_json(LOGIN_ATTEMPTS_FILE, data)


def check_lockout(user_id: str):
    """
    로그인 시도가 가능한 상태인지 확인합니다.
    반환: (locked: bool, remaining_seconds: int)
    """
    attempts = _load_login_attempts()
    record = attempts.get(user_id)

    if not record:
        return False, 0

    locked_until = record.get("locked_until")

    if not locked_until:
        return False, 0

    remaining = locked_until - time.time()

    if remaining <= 0:
        def _apply(data):
            existing = data.get(user_id)
            if existing:
                existing["locked_until"] = None
                existing["failed_count"] = 0
                data[user_id] = existing
            return data

        storage.update_json(LOGIN_ATTEMPTS_FILE, {}, _apply)
        return False, 0

    return True, int(remaining)


def register_failed_login(user_id: str):
    """로그인 실패를 기록하고, 한도를 넘으면 잠금 처리합니다."""
    _ensure_data_dir()

    def _apply(attempts):
        record = attempts.get(
            user_id,
            {"failed_count": 0, "locked_until": None},
        )
        record["failed_count"] = int(record.get("failed_count", 0)) + 1

        if record["failed_count"] >= LOGIN_MAX_ATTEMPTS:
            record["locked_until"] = time.time() + LOGIN_LOCKOUT_SECONDS

        attempts[user_id] = record
        return attempts

    storage.update_json(LOGIN_ATTEMPTS_FILE, {}, _apply)


def register_successful_login(user_id: str):
    """로그인 성공 시 실패 기록을 초기화합니다."""
    _ensure_data_dir()

    def _apply(attempts):
        if user_id in attempts:
            attempts[user_id] = {"failed_count": 0, "locked_until": None}
        return attempts

    storage.update_json(LOGIN_ATTEMPTS_FILE, {}, _apply)


def _today_str() -> str:
    return time.strftime("%Y-%m-%d")


def _load_usage() -> dict:
    return _load_json(USAGE_FILE, {})


def _save_usage(data: dict):
    _save_json(USAGE_FILE, data)


def get_remaining_calls(owner_key: str, daily_limit: int = DAILY_AI_CALL_LIMIT) -> int:
    """오늘 남은 AI 호출 가능 횟수를 조회합니다 (호출 횟수를 증가시키지 않음)."""
    usage = _load_usage()
    record = usage.get(owner_key)

    if not record or record.get("date") != _today_str():
        return daily_limit

    return max(0, daily_limit - int(record.get("count", 0)))


def try_consume_call(owner_key: str, daily_limit: int = DAILY_AI_CALL_LIMIT):
    """
    AI 호출 한 번을 사용 처리합니다.
    반환: (allowed: bool, remaining_after: int)
    한도를 넘었으면 (False, 0)을 반환하고 카운트를 늘리지 않습니다.

    파일 락으로 read-modify-write를 보호하므로, 여러 요청이 동시에
    들어와도 한도를 초과해 통과하는 일이 없습니다.
    """
    _ensure_data_dir()
    today = _today_str()
    outcome = {"allowed": False, "remaining": 0}

    def _apply(usage):
        record = usage.get(owner_key)

        if not record or record.get("date") != today:
            record = {"date": today, "count": 0}

        if record["count"] >= daily_limit:
            usage[owner_key] = record
            outcome["allowed"] = False
            outcome["remaining"] = 0
            return usage

        record["count"] += 1
        usage[owner_key] = record
        outcome["allowed"] = True
        outcome["remaining"] = daily_limit - record["count"]
        return usage

    storage.update_json(USAGE_FILE, {}, _apply)
    return outcome["allowed"], outcome["remaining"]


def validate_pdf_upload(uploaded_file, max_size_mb: int = MAX_PDF_SIZE_MB):
    """
    업로드된 파일이 안전하게 처리 가능한 PDF인지 검사합니다.
    반환: (ok: bool, error_message: str | None)
    """
    filename = str(getattr(uploaded_file, "name", ""))

    if not filename.lower().endswith(".pdf"):
        return False, f"'{filename}'은(는) PDF 파일이 아닙니다."

    size_bytes = getattr(uploaded_file, "size", None)

    if size_bytes is None:
        size_bytes = len(uploaded_file.getvalue())

    max_bytes = max_size_mb * 1024 * 1024

    if size_bytes > max_bytes:
        size_mb = size_bytes / (1024 * 1024)
        return False, (
            f"'{filename}' 파일이 너무 큽니다 "
            f"({size_mb:.1f}MB, 최대 {max_size_mb}MB)."
        )

    if size_bytes == 0:
        return False, f"'{filename}' 파일이 비어 있습니다."

    header = uploaded_file.getvalue()[:5]

    if header != b"%PDF-":
        return False, (
            f"'{filename}'의 내용이 올바른 PDF 형식이 아닙니다."
        )

    return True, None


def configure_api_keys():
    """
    Streamlit Cloud의 st.secrets에 GOOGLE_API_KEY가 있으면 환경변수로 반영합니다.
    없으면 로컬 .env(환경변수)에 이미 설정된 값을 그대로 둡니다.
    두 곳 모두에 키가 없어도 여기서 앱을 중단시키지 않고,
    실제 API 호출 시점에 오류 메시지로 알립니다.

    GEMINI_API_KEY 이름으로 넣은 경우도 함께 지원합니다.
    """
    try:
        secret_key = (
            st.secrets.get("GOOGLE_API_KEY")
            or st.secrets.get("GEMINI_API_KEY")
        )
    except Exception:
        secret_key = None

    if secret_key:
        os.environ["GOOGLE_API_KEY"] = secret_key
    elif os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]


configure_openai_api_key = configure_api_keys
