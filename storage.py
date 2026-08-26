import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

try:
    import fcntl
    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False

try:
    import msvcrt
    _HAS_MSVCRT = True
except ImportError:
    _HAS_MSVCRT = False


def is_cloud_deployment() -> bool:
    """
    Streamlit Community Cloud에서 실행 중인지 추정합니다.
    (플랫폼이 설정하는 환경변수/경로 힌트를 활용. 정확한 공식 API는 아니며
     로컬과 클라우드의 동작을 다르게 하고 싶을 때 참고용으로 사용합니다.)
    """
    hints = (
        os.environ.get("STREAMLIT_RUNTIME_ENV"),
        os.environ.get("STREAMLIT_SERVER_HEADLESS"),
    )

    if any(hint for hint in hints):
        return True

    home = os.environ.get("HOME", "")
    return "/home/appuser" in home or "/mount/src" in os.getcwd()


@contextmanager
def _locked_file(lock_path: Path, timeout: float = 10.0):
    """
    lock_path에 대한 배타적(exclusive) 파일 락을 획득합니다.
    같은 서버 프로세스 내 여러 세션이 동시에 같은 JSON을 수정할 때
    read-modify-write 구간이 겹치지 않도록 보호합니다.

    락을 지원하지 않는 환경이면 락 없이 그대로 진행합니다(best-effort).
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = open(lock_path, "a+")

    try:
        if _HAS_FCNTL:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        elif _HAS_MSVCRT:
            deadline = time.time() + timeout
            while True:
                try:
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
                    break
                except OSError:
                    if time.time() >= deadline:
                        break
                    time.sleep(0.1)

        yield
    finally:
        try:
            if _HAS_FCNTL:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            elif _HAS_MSVCRT:
                try:
                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
        finally:
            lock_file.close()


def read_json(path: Path, default):
    """
    JSON 파일을 읽습니다. 파일이 없거나 손상되었으면 default(깊은 복사)를 돌려줍니다.
    읽기는 락 없이 수행합니다(원자적 쓰기로 파일이 항상 완전한 상태이기 때문).
    """
    path = Path(path)

    if not path.exists():
        return json.loads(json.dumps(default))

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return json.loads(json.dumps(default))


def write_json(path: Path, data) -> None:
    """
    임시 파일에 쓴 뒤 원자적으로 교체(replace)합니다.
    쓰기 도중 앱이 중단되어도 원본 파일이 반쪽짜리로 깨지지 않습니다.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_file = path.with_suffix(path.suffix + ".tmp")
    temp_file.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_file.replace(path)


def update_json(path: Path, default, mutate_fn):
    """
    동시성 안전한 read-modify-write.

    1) 파일 락을 획득한 뒤
    2) 현재 값을 읽고
    3) mutate_fn(data)로 수정한 결과를
    4) 원자적으로 저장합니다.

    mutate_fn은 수정된 dict/list를 반환해야 하며, 반환값이 None이면
    전달받은 data를 그대로 저장합니다(제자리 수정 지원).

    반환: 최종 저장된 데이터.
    """
    path = Path(path)
    lock_path = path.with_suffix(path.suffix + ".lock")

    with _locked_file(lock_path):
        data = read_json(path, default)
        result = mutate_fn(data)

        if result is None:
            result = data

        write_json(path, result)
        return result
