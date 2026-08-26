import hashlib
import json
import re
import uuid
from pathlib import Path
import pymupdf
import streamlit as st

import security
import storage
from app_logging import get_logger, log_event, log_exception
from rag_module import (
    process_pdfs_and_get_chain,
    prune_vector_cache as _prune_vector_cache_dir,
)

security.configure_api_keys()

logger = get_logger()


STUDENT_ANSWER_LANGUAGES = {
    "한국어": "답변은 자연스럽고 이해하기 쉬운 한국어로 작성하세요.",
    "English": (
        "Write the entire answer in clear, natural English. "
        "Explain technical terms in student-friendly language."
    ),
    "한국어 + 영어 핵심용어": (
        "답변 본문은 자연스러운 한국어로 작성하고, 핵심 전문용어는 "
        "영어를 괄호 안에 함께 표기하세요. "
        "예: 단면계수(section modulus)."
    ),
}


st.set_page_config(
    page_title="AI 학습지원 플랫폼",
    layout="wide"
)


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "role" not in st.session_state:
    st.session_state.role = None

if "username" not in st.session_state:
    st.session_state.username = None

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "rag_filename" not in st.session_state:
    st.session_state.rag_filename = None

if "rag_result" not in st.session_state:
    st.session_state.rag_result = None

if "document_public" not in st.session_state:
    st.session_state.document_public = False

if "rag_chains" not in st.session_state:
    st.session_state.rag_chains = {}

if "rag_chains_order" not in st.session_state:
    st.session_state.rag_chains_order = []

if "rag_selection" not in st.session_state:
    st.session_state.rag_selection = ()

if "professor_panel_visible" not in st.session_state:
    st.session_state.professor_panel_visible = True

if "student_panel_visible" not in st.session_state:
    st.session_state.student_panel_visible = True

if "professor_chat_history" not in st.session_state:
    st.session_state.professor_chat_history = []

if "student_chat_history" not in st.session_state:
    st.session_state.student_chat_history = []

if "student_quiz" not in st.session_state:
    st.session_state.student_quiz = []

if "student_quiz_feedback" not in st.session_state:
    st.session_state.student_quiz_feedback = None

if "student_quiz_version" not in st.session_state:
    st.session_state.student_quiz_version = 0

if "student_excerpt_index" not in st.session_state:
    st.session_state.student_excerpt_index = 0

if "pending_document_delete" not in st.session_state:
    st.session_state.pending_document_delete = None

if "chat_context_role" not in st.session_state:
    st.session_state.chat_context_role = None


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
VECTOR_CACHE_DIR = DATA_DIR / "vector_cache"
DOCUMENTS_FILE = DATA_DIR / "documents.json"
CHAT_SESSIONS_FILE = DATA_DIR / "chat_sessions.json"

MAX_RAG_CHAINS_IN_MEMORY = 3

MAX_CHAT_SESSIONS_PER_ROLE = 20


def ensure_document_storage():
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    VECTOR_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not DOCUMENTS_FILE.exists():
        storage.write_json(DOCUMENTS_FILE, {"documents": {}})

    if not CHAT_SESSIONS_FILE.exists():
        storage.write_json(CHAT_SESSIONS_FILE, {"users": {}})


def load_documents():
    ensure_document_storage()

    data = storage.read_json(DOCUMENTS_FILE, {"documents": {}})
    documents = data.get("documents", {})

    if not isinstance(documents, dict):
        documents = {}

    return {
        filename: metadata
        for filename, metadata in documents.items()
        if (PDF_DIR / filename).is_file()
    }


def save_documents(documents):
    ensure_document_storage()
    storage.write_json(DOCUMENTS_FILE, {"documents": documents})


def load_saved_chat_sessions():
    """역할별 채팅 목록을 디스크에서 읽습니다."""
    ensure_document_storage()

    data = storage.read_json(CHAT_SESSIONS_FILE, {"users": {}})
    users = data.get("users", {})
    return users if isinstance(users, dict) else {}


def save_saved_chat_sessions(users):
    """역할별 채팅 목록을 임시 파일을 거쳐 안전하게 저장합니다."""
    ensure_document_storage()
    storage.write_json(CHAT_SESSIONS_FILE, {"users": users})


def save_uploaded_pdfs(uploaded_files):
    saved_files = []
    rejected_files = []
    valid_uploads = []

    for uploaded_file in uploaded_files:
        is_valid, error_message = security.validate_pdf_upload(
            uploaded_file
        )

        if not is_valid:
            rejected_files.append(error_message)
            continue

        filename = Path(uploaded_file.name).name

        if not filename.lower().endswith(".pdf"):
            continue

        (PDF_DIR / filename).write_bytes(
            uploaded_file.getvalue()
        )
        valid_uploads.append((filename, uploaded_file.size))

        stale_keys = [
            cache_key
            for cache_key in st.session_state.rag_chains
            if filename in (
                cache_key
                if isinstance(cache_key, tuple)
                else (cache_key,)
            )
        ]

        for cache_key in stale_keys:
            _remove_rag_chain(cache_key)

        if filename in st.session_state.rag_selection:
            st.session_state.rag_chain = None
            st.session_state.rag_filename = None
            st.session_state.rag_selection = ()
            st.session_state.rag_result = None

        saved_files.append(filename)

    def _apply(data):
        documents = data.get("documents", {})

        if not isinstance(documents, dict):
            documents = {}

        for filename, size in valid_uploads:
            previous = documents.get(filename, {})
            documents[filename] = {
                "public": bool(previous.get("public", False)),
                "size": size,
            }

        data["documents"] = documents
        return data

    ensure_document_storage()
    storage.update_json(DOCUMENTS_FILE, {"documents": {}}, _apply)
    return saved_files, rejected_files


def set_document_public(filename, is_public):
    def _apply(data):
        documents = data.get("documents", {})

        if not isinstance(documents, dict) or filename not in documents:
            return data

        documents[filename]["public"] = bool(is_public)
        data["documents"] = documents
        return data

    ensure_document_storage()
    storage.update_json(DOCUMENTS_FILE, {"documents": {}}, _apply)


def delete_document(filename):
    """저장된 PDF와 관련 선택·RAG 상태를 함께 삭제합니다."""
    safe_filename = Path(str(filename)).name

    if safe_filename != str(filename):
        return False

    documents = load_documents()

    if safe_filename not in documents:
        return False

    pdf_path = PDF_DIR / safe_filename

    if pdf_path.is_file():
        pdf_path.unlink()

    def _apply(data):
        docs = data.get("documents", {})

        if isinstance(docs, dict):
            docs.pop(safe_filename, None)

        data["documents"] = docs
        return data

    ensure_document_storage()
    storage.update_json(DOCUMENTS_FILE, {"documents": {}}, _apply)

    stale_keys = [
        cache_key
        for cache_key in st.session_state.rag_chains
        if safe_filename in (
            cache_key
            if isinstance(cache_key, tuple)
            else (cache_key,)
        )
    ]

    for cache_key in stale_keys:
        _remove_rag_chain(cache_key)

    try:
        prune_vector_cache()
    except Exception as e:
        log_exception("벡터 캐시 정리 실패", e)

    for selection_key in (
        "professor_selected_documents",
        "student_selected_documents",
    ):
        saved_selection = st.session_state.get(selection_key, [])
        st.session_state[selection_key] = [
            selected_filename
            for selected_filename in saved_selection
            if selected_filename != safe_filename
        ]

    if safe_filename in st.session_state.rag_selection:
        st.session_state.rag_chain = None
        st.session_state.rag_filename = None
        st.session_state.rag_selection = ()
        st.session_state.rag_result = None
        st.session_state.student_quiz = []
        st.session_state.student_quiz_feedback = None

    st.session_state.pop("delete_document_target", None)
    st.session_state.pending_document_delete = None
    return True


def _touch_rag_chain(selection):
    """selection을 최근 사용 목록의 맨 뒤로 옮깁니다(LRU 갱신)."""
    order = st.session_state.rag_chains_order

    if selection in order:
        order.remove(selection)

    order.append(selection)


def _remove_rag_chain(cache_key):
    """RAG 체인 하나를 캐시와 LRU 순서 목록에서 함께 제거합니다."""
    st.session_state.rag_chains.pop(cache_key, None)
    order = st.session_state.rag_chains_order

    if cache_key in order:
        order.remove(cache_key)


def prune_vector_cache():
    """
    디스크 벡터 캐시(data/vector_cache)의 폴더 수를 제한합니다.

    캐시 폴더는 'PDF 조합'별 내용 해시라서 현재 PDF만으로 유효 해시 전체를
    정확히 재현하긴 어렵습니다. 그래서 여기서는 개수 상한만 적용해
    가장 오래된 캐시부터 정리합니다(정확도보다 저장공간 안정성 우선).
    잘못 지워지더라도 다음 선택 시 자동으로 다시 임베딩되므로 안전합니다.
    """
    _prune_vector_cache_dir(
        VECTOR_CACHE_DIR,
        keep_hashes=None,
        max_entries=20,
    )


def _evict_rag_chains_if_needed():
    """
    세션 메모리의 RAG 체인 수가 한도를 넘으면 가장 오래 전에 사용한 것부터
    정리합니다. 현재 활성 selection은 정리 대상에서 제외합니다.
    """
    order = st.session_state.rag_chains_order
    active = st.session_state.rag_selection

    while len(st.session_state.rag_chains) > MAX_RAG_CHAINS_IN_MEMORY:
        victim = None

        for candidate in order:
            if candidate != active and candidate in st.session_state.rag_chains:
                victim = candidate
                break

        if victim is None:
            break

        st.session_state.rag_chains.pop(victim, None)

        if victim in order:
            order.remove(victim)

        log_event(f"RAG 체인 메모리 정리(LRU): {', '.join(victim)}")


def activate_documents(filenames):
    selection = tuple(sorted(filenames))

    if not selection:
        st.session_state.rag_chain = None
        st.session_state.rag_filename = None
        st.session_state.rag_selection = ()
        return

    if selection not in st.session_state.rag_chains:
        pdf_paths = [
            PDF_DIR / filename
            for filename in selection
        ]

        missing_files = [
            path.name
            for path in pdf_paths
            if not path.is_file()
        ]

        if missing_files:
            raise FileNotFoundError(
                "저장된 PDF를 찾을 수 없습니다: "
                + ", ".join(missing_files)
            )

        progress_placeholder = st.empty()
        progress_bar = progress_placeholder.progress(
            0,
            text="PDF 임베딩 준비 중...",
        )

        def _update_embedding_progress(done, total):
            ratio = (done / total) if total else 1.0
            progress_bar.progress(
                min(max(ratio, 0.0), 1.0),
                text=f"PDF 임베딩 처리 중... ({done}/{total} 청크)",
            )

        try:
            st.session_state.rag_chains[selection] = (
                process_pdfs_and_get_chain(
                    [str(path) for path in pdf_paths],
                    cache_dir=VECTOR_CACHE_DIR,
                    progress_callback=_update_embedding_progress,
                )
            )
        finally:
            progress_placeholder.empty()

    if st.session_state.rag_selection != selection:
        st.session_state.rag_result = None

    st.session_state.rag_chain = (
        st.session_state.rag_chains[selection]
    )
    st.session_state.rag_selection = selection
    st.session_state.rag_filename = ", ".join(selection)

    _touch_rag_chain(selection)
    _evict_rag_chains_if_needed()


@st.cache_data(show_spinner=False)
def render_pdf_page(pdf_path, page_number, modified_time_ns):
    """PDF의 특정 페이지를 브라우저 표시용 PNG로 변환합니다."""
    del modified_time_ns

    document = pymupdf.open(pdf_path)

    try:
        page_index = int(page_number) - 1

        if page_index < 0 or page_index >= len(document):
            raise ValueError(
                f"PDF에 {page_number}페이지가 없습니다."
            )

        page = document[page_index]
        pixmap = page.get_pixmap(
            dpi=144,
            alpha=False
        )
        return pixmap.tobytes("png")
    finally:
        document.close()


@st.cache_data(show_spinner=False)
def load_pdf_bytes(pdf_path, modified_time_ns):
    """다운로드할 PDF 원본을 캐시해서 읽습니다."""
    del modified_time_ns
    return Path(pdf_path).read_bytes()


def display_grouped_sources(sources):
    """출처를 파일별로 묶어 페이지를 한 줄로 표시합니다."""
    grouped_sources = {}
    plain_sources = []

    for source in sources:
        if not isinstance(source, dict):
            plain_sources.append(str(source))
            continue

        raw_file_name = source.get(
            "file",
            st.session_state.rag_filename or "강의자료"
        )
        file_name = Path(str(raw_file_name)).name
        page = source.get("page")

        grouped_sources.setdefault(file_name, set())

        try:
            grouped_sources[file_name].add(int(page))
        except (TypeError, ValueError):
            if page not in (None, "", "?"):
                grouped_sources[file_name].add(str(page))

    for file_name, pages in grouped_sources.items():
        numeric_pages = sorted(
            page for page in pages
            if isinstance(page, int)
        )
        text_pages = sorted(
            str(page) for page in pages
            if not isinstance(page, int)
        )
        page_values = [
            *(str(page) for page in numeric_pages),
            *text_pages,
        ]

        if page_values:
            st.write(
                f"📄 {file_name} — p.{', '.join(page_values)}"
            )
        else:
            st.write(f"📄 {file_name}")

    for source in plain_sources:
        st.write(f"📄 {source}")


def change_student_excerpt(step, excerpt_count):
    """학생 근거 발췌 카드의 현재 위치를 이동합니다."""
    current_index = st.session_state.student_excerpt_index
    st.session_state.student_excerpt_index = max(
        0,
        min(current_index + step, excerpt_count - 1),
    )


def display_student_source_excerpts(sources):
    """AI가 참고한 PDF 문단을 한 장씩 넘기는 카드로 표시합니다."""
    documents = load_documents()
    excerpts = []
    seen = set()

    for source in sources:
        if not isinstance(source, dict):
            continue

        excerpt = str(source.get("excerpt", "")).strip()

        if not excerpt:
            continue

        file_name = Path(
            str(source.get("file", ""))
        ).name
        page = source.get("page", "?")
        metadata = documents.get(file_name)
        pdf_path = PDF_DIR / file_name

        if (
            metadata is None
            or not metadata.get("public", False)
            or not pdf_path.is_file()
        ):
            continue

        key = (file_name, page, excerpt)

        if key not in seen:
            seen.add(key)
            excerpts.append({
                "file": file_name,
                "page": page,
                "excerpt": excerpt,
            })

    if not excerpts:
        return

    current_index = max(
        0,
        min(
            st.session_state.student_excerpt_index,
            len(excerpts) - 1,
        ),
    )
    st.session_state.student_excerpt_index = current_index
    current_excerpt = excerpts[current_index]

    st.write("**AI가 참고한 강의자료 발췌**")
    previous_col, status_col, next_col = st.columns([1, 2, 1])

    with previous_col:
        st.button(
            "◀ 이전 근거",
            key="previous_student_excerpt",
            disabled=(current_index == 0),
            use_container_width=True,
            on_click=change_student_excerpt,
            args=(-1, len(excerpts)),
        )

    with status_col:
        st.markdown(
            f"<p style='text-align:center;'>"
            f"근거 {current_index + 1} / {len(excerpts)}"
            f"</p>",
            unsafe_allow_html=True,
        )

    with next_col:
        st.button(
            "다음 근거 ▶",
            key="next_student_excerpt",
            disabled=(current_index == len(excerpts) - 1),
            use_container_width=True,
            on_click=change_student_excerpt,
            args=(1, len(excerpts)),
        )

    with st.container(height=720, border=True):
        st.markdown(
            f"**📄 {current_excerpt['file']} "
            f"— p.{current_excerpt['page']}**"
        )

        pdf_path = PDF_DIR / current_excerpt["file"]

        try:
            page_number = int(current_excerpt["page"])
            modified_time_ns = pdf_path.stat().st_mtime_ns
            page_image = render_pdf_page(
                str(pdf_path),
                page_number,
                modified_time_ns,
            )
            st.image(
                page_image,
                caption=(
                    f"{current_excerpt['file']} "
                    f"- {page_number}페이지"
                ),
                width="stretch",
            )
        except (TypeError, ValueError, OSError, RuntimeError):
            st.caption(
                "PDF 원문 페이지 미리보기를 표시하지 못했습니다."
            )

        st.caption("AI가 참고한 부분")
        st.write(current_excerpt["excerpt"])


def chat_session_keys(role):
    """역할별 채팅 세션 저장 키를 반환합니다."""
    return f"{role}_chat_sessions", f"{role}_active_chat_id"


def create_empty_chat_session(role):
    """새 채팅에 필요한 초기 상태를 만듭니다."""
    return {
        "id": f"{role}_{uuid.uuid4().hex[:12]}",
        "title": "새 채팅",
        "history": [],
        "rag_result": None,
        "student_quiz": [],
        "student_quiz_feedback": None,
    }


def chat_owner_key(role):
    """로그인한 계정과 역할을 채팅 저장소의 고유 키로 만듭니다."""
    username = st.session_state.get("username") or role
    return f"{role}:{username}"


def persist_role_chat_sessions(role):
    """현재 역할의 채팅 목록과 활성 채팅을 디스크에 저장합니다."""
    sessions_key, active_key = chat_session_keys(role)

    if sessions_key not in st.session_state:
        return

    owner_key = chat_owner_key(role)
    payload = {
        "sessions": st.session_state[sessions_key],
        "active_id": st.session_state.get(active_key),
    }

    def _apply(data):
        users = data.get("users", {})

        if not isinstance(users, dict):
            users = {}

        users[owner_key] = payload
        data["users"] = users
        return data

    ensure_document_storage()
    storage.update_json(CHAT_SESSIONS_FILE, {"users": {}}, _apply)


def load_chat_session(role, history_key, chat_id):
    """선택한 채팅의 대화·답변 상태를 현재 화면으로 불러옵니다."""
    sessions_key, active_key = chat_session_keys(role)
    sessions = st.session_state[sessions_key]
    session = sessions[chat_id]

    st.session_state[active_key] = chat_id
    st.session_state[history_key] = list(session.get("history", []))
    st.session_state.rag_result = session.get("rag_result")
    st.session_state.student_excerpt_index = 0

    if role == "student":
        st.session_state.student_quiz = list(
            session.get("student_quiz", [])
        )
        st.session_state.student_quiz_feedback = session.get(
            "student_quiz_feedback"
        )

    st.session_state.chat_context_role = role
    persist_role_chat_sessions(role)


def ensure_chat_sessions(role, history_key):
    """역할별 채팅 목록과 현재 채팅을 준비합니다."""
    sessions_key, active_key = chat_session_keys(role)

    if sessions_key not in st.session_state:
        saved_user_chats = load_saved_chat_sessions().get(
            chat_owner_key(role),
            {},
        )
        saved_sessions = saved_user_chats.get("sessions", {})
        saved_active_id = saved_user_chats.get("active_id")

        if isinstance(saved_sessions, dict) and saved_sessions:
            st.session_state[sessions_key] = saved_sessions
            st.session_state[active_key] = saved_active_id
        else:
            first_session = create_empty_chat_session(role)
            first_session["history"] = list(
                st.session_state.get(history_key, [])
            )
            st.session_state[sessions_key] = {
                first_session["id"]: first_session
            }
            st.session_state[active_key] = first_session["id"]

    sessions = st.session_state[sessions_key]
    active_id = st.session_state.get(active_key)

    if active_id not in sessions:
        first_session = create_empty_chat_session(role)
        sessions[first_session["id"]] = first_session
        active_id = first_session["id"]
        st.session_state[active_key] = active_id

    if st.session_state.chat_context_role != role:
        load_chat_session(role, history_key, active_id)


def sync_active_chat_session(role, history_key):
    """현재 화면 상태를 활성 채팅에 저장합니다."""
    sessions_key, active_key = chat_session_keys(role)

    if sessions_key not in st.session_state:
        return

    active_id = st.session_state.get(active_key)
    sessions = st.session_state[sessions_key]

    if active_id not in sessions:
        return

    session = sessions[active_id]
    session["history"] = list(st.session_state.get(history_key, []))
    session["rag_result"] = st.session_state.rag_result

    if role == "student":
        session["student_quiz"] = list(st.session_state.student_quiz)
        session["student_quiz_feedback"] = (
            st.session_state.student_quiz_feedback
        )

    persist_role_chat_sessions(role)


def _prune_chat_sessions(role):
    """
    역할당 채팅방 수가 한도를 넘으면 오래된 채팅부터 정리합니다.
    현재 활성 채팅은 정리 대상에서 제외합니다.
    dict의 삽입 순서를 활용해 앞쪽(오래된)부터 제거합니다.
    """
    sessions_key, active_key = chat_session_keys(role)
    sessions = st.session_state.get(sessions_key, {})
    active_id = st.session_state.get(active_key)

    if len(sessions) <= MAX_CHAT_SESSIONS_PER_ROLE:
        return

    removable_ids = [
        chat_id
        for chat_id in sessions
        if chat_id != active_id
    ]

    remove_count = len(sessions) - MAX_CHAT_SESSIONS_PER_ROLE

    for chat_id in removable_ids[:remove_count]:
        sessions.pop(chat_id, None)

    log_event(
        f"오래된 채팅 정리: role={role}, "
        f"{remove_count}개 제거"
    )


def start_new_chat(role, history_key):
    """현재 채팅을 보관하고 빈 채팅을 활성화합니다."""
    ensure_chat_sessions(role, history_key)
    sync_active_chat_session(role, history_key)

    sessions_key, active_key = chat_session_keys(role)
    new_session = create_empty_chat_session(role)
    st.session_state[sessions_key][new_session["id"]] = new_session
    st.session_state[active_key] = new_session["id"]

    _prune_chat_sessions(role)

    load_chat_session(role, history_key, new_session["id"])


def switch_chat_session(role, history_key, chat_id):
    """현재 채팅을 보관한 뒤 다른 채팅으로 이동합니다."""
    sync_active_chat_session(role, history_key)
    load_chat_session(role, history_key, chat_id)


def display_chat_session_sidebar(role, history_key):
    """자료 패널에 새 채팅 버튼과 최근 채팅 목록을 표시합니다."""
    ensure_chat_sessions(role, history_key)
    sessions_key, active_key = chat_session_keys(role)

    st.subheader("💬 채팅")

    if st.button(
        "＋ 새 채팅",
        key=f"{role}_new_chat",
        use_container_width=True,
        type="primary",
    ):
        start_new_chat(role, history_key)
        st.rerun()

    st.caption("최근 채팅")
    sessions = st.session_state[sessions_key]
    active_id = st.session_state[active_key]

    with st.container(height=220, border=True):
        for chat_id, session in reversed(list(sessions.items())):
            title = session.get("title", "새 채팅")
            label = f"● {title}" if chat_id == active_id else title

            if st.button(
                label,
                key=f"{role}_open_chat_{chat_id}",
                use_container_width=True,
                disabled=chat_id == active_id,
            ):
                switch_chat_session(role, history_key, chat_id)
                st.rerun()


def append_chat_history(history_key, user_message, result):
    """질문과 AI 응답을 역할별 대화 기록에 추가합니다."""
    if isinstance(result, dict):
        answer = result.get(
            "answer",
            "답변을 생성하지 못했습니다."
        )
        has_evidence = result.get("has_evidence", True)
    else:
        answer = str(result)
        has_evidence = True

    st.session_state[history_key].append({
        "user": user_message,
        "assistant": answer,
        "has_evidence": has_evidence,
    })

    role = "professor" if history_key.startswith("professor") else "student"
    sessions_key, active_key = chat_session_keys(role)

    if sessions_key in st.session_state:
        active_id = st.session_state.get(active_key)
        session = st.session_state[sessions_key].get(active_id)

        if session and session.get("title") == "새 채팅":
            compact_title = " ".join(str(user_message).split())
            session["title"] = (
                compact_title[:24] + "…"
                if len(compact_title) > 24
                else compact_title
            ) or "새 채팅"

        sync_active_chat_session(role, history_key)


def run_streaming_answer(
    history_key,
    display_user_message,
    question_for_chain,
    spinner_text,
    answer_instruction="",
):
    """
    RAGChain.stream()을 이용해 답변을 실시간으로 출력한 뒤,
    완성된 답변을 대화 기록에 저장합니다.

    - 검색(근거 확보)까지는 스피너를 보여주고,
    - 이후 토큰이 흘러나오는 대로 st.write_stream으로 화면에 즉시 표시합니다.
    - 근거가 없으면 경고 스타일로 출력합니다.

    반환: 성공 여부(bool). 실패 시 오류 로그를 남기고 False를 반환합니다.
    """
    chain = st.session_state.rag_chain

    try:
        with st.spinner(spinner_text, show_time=True):
            stream_gen, meta = chain.stream(
                question_for_chain,
                answer_instruction=answer_instruction,
            )
    except Exception as e:
        log_exception("AI 스트리밍 준비 실패", e)
        st.error(
            "AI 답변 생성 중 오류가 발생했습니다. "
            "잠시 후 다시 시도해주세요."
        )
        return False

    has_evidence = meta.get("has_evidence", True)

    try:
        with st.chat_message("assistant"):
            if has_evidence:
                answer_text = st.write_stream(stream_gen)
            else:
                answer_text = "".join(stream_gen)
                st.warning(answer_text)
    except Exception as e:
        log_exception("AI 스트리밍 출력 실패", e)
        st.error(
            "AI 답변 생성 중 오류가 발생했습니다. "
            "잠시 후 다시 시도해주세요."
        )
        return False

    result = {
        "answer": answer_text,
        "sources": meta.get("sources", []),
        "has_evidence": has_evidence,
    }
    st.session_state.rag_result = result
    st.session_state.student_excerpt_index = 0
    append_chat_history(history_key, display_user_message, result)

    st.rerun()
    return True


def display_chat_history(history_key, clear_button_key):
    """대화 기록을 고정 높이의 스크롤 영역으로 표시합니다."""
    title_col, clear_col = st.columns([4, 1])

    with title_col:
        st.subheader("💬 대화 기록")

    with clear_col:
        if st.button(
            "기록 지우기",
            key=clear_button_key,
            use_container_width=True,
        ):
            st.session_state[history_key] = []
            st.session_state.rag_result = None
            role = (
                "professor"
                if history_key.startswith("professor")
                else "student"
            )
            sync_active_chat_session(role, history_key)
            st.rerun()

    history = st.session_state[history_key]

    with st.container(
        height=420,
        border=True,
        autoscroll=True,
    ):
        if not history:
            st.info("질문하면 대화가 여기에 차례대로 쌓입니다.")

        for message in history:
            with st.chat_message("user"):
                st.markdown(message["user"])

            with st.chat_message("assistant"):
                if message.get("has_evidence", True):
                    st.markdown(message["assistant"])
                else:
                    st.warning(message["assistant"])


def normalize_quiz_answer(value):
    """단답형 채점을 위해 공백과 문장부호를 제거합니다."""
    return re.sub(
        r"[\W_]+",
        "",
        str(value or "").lower(),
        flags=re.UNICODE,
    )


def display_quiz_source_preview(file_name, page_number, toggle_key):
    """퀴즈 해설 안에서 공개된 PDF의 근거 페이지를 표시합니다."""
    documents = load_documents()
    safe_file_name = Path(str(file_name)).name
    metadata = documents.get(safe_file_name)
    pdf_path = PDF_DIR / safe_file_name

    if (
        metadata is None
        or not metadata.get("public", False)
        or not pdf_path.is_file()
    ):
        st.caption("현재 열람할 수 없는 근거 페이지입니다.")
        return

    try:
        page_number = int(page_number)
    except (TypeError, ValueError):
        st.caption("근거 페이지 번호를 확인할 수 없습니다.")
        return

    if st.toggle(
        "📄 근거 페이지 보기",
        key=toggle_key,
    ):
        try:
            modified_time_ns = pdf_path.stat().st_mtime_ns
            page_image = render_pdf_page(
                str(pdf_path),
                page_number,
                modified_time_ns,
            )
            st.image(
                page_image,
                caption=f"{safe_file_name} - {page_number}페이지",
                width="stretch",
            )
        except Exception as e:
            log_exception("퀴즈 근거 페이지 렌더링 실패", e)
            st.warning(
                "근거 페이지를 표시하지 못했습니다."
            )


def display_student_quiz():
    """학생용 퀴즈 폼과 자동 채점 결과를 표시합니다."""
    quiz = st.session_state.student_quiz

    if not quiz:
        return

    st.subheader("📝 강의자료 복습 퀴즈")
    st.caption(
        "모든 문제에 답한 뒤 제출하면 점수와 해설을 확인할 수 있습니다."
    )

    version = st.session_state.student_quiz_version

    with st.form(
        f"student_quiz_form_{version}",
        clear_on_submit=False,
    ):
        submitted_answers = []

        for index, item in enumerate(quiz, start=1):
            question_type = item.get("question_type")
            type_label = "OX" if question_type == "ox" else "단답형"

            st.markdown(
                f"**{index}. [{type_label}] {item.get('question', '')}**"
            )

            if question_type == "ox":
                answer = st.radio(
                    f"{index}번 답",
                    ["O", "X"],
                    index=None,
                    horizontal=True,
                    key=f"quiz_{version}_{index}_ox",
                    label_visibility="collapsed",
                )
            else:
                answer = st.text_input(
                    f"{index}번 답",
                    placeholder="짧게 입력하세요",
                    key=f"quiz_{version}_{index}_short",
                    label_visibility="collapsed",
                )

            submitted_answers.append(answer)
            st.divider()

        submit_quiz = st.form_submit_button(
            "답안 제출하고 채점하기",
            type="primary",
            use_container_width=True,
        )

    if submit_quiz:
        feedback_items = []
        score = 0

        for item, submitted_answer in zip(
            quiz,
            submitted_answers,
        ):
            question_type = item.get("question_type")
            correct_answer = str(item.get("answer", ""))

            if question_type == "ox":
                is_correct = (
                    str(submitted_answer or "").upper()
                    == correct_answer.upper()
                )
            else:
                user_answer = normalize_quiz_answer(
                    submitted_answer
                )
                accepted_answers = [
                    correct_answer,
                    *item.get("accepted_answers", []),
                ]
                normalized_answers = {
                    normalize_quiz_answer(answer)
                    for answer in accepted_answers
                    if normalize_quiz_answer(answer)
                }
                is_correct = bool(user_answer) and any(
                    user_answer == accepted
                    or accepted in user_answer
                    for accepted in normalized_answers
                )

            if is_correct:
                score += 1

            feedback_items.append({
                "is_correct": is_correct,
                "submitted_answer": submitted_answer or "미응답",
                "correct_answer": correct_answer,
                "explanation": item.get("explanation", ""),
                "source_file": item.get("source_file", "강의자료"),
                "page": item.get("page", "?"),
            })

        st.session_state.student_quiz_feedback = {
            "score": score,
            "total": len(quiz),
            "items": feedback_items,
        }
        sync_active_chat_session(
            "student",
            "student_chat_history",
        )

    feedback = st.session_state.student_quiz_feedback

    if feedback:
        score = feedback["score"]
        total = feedback["total"]
        st.success(f"채점 결과: {score} / {total}점")

        for index, item in enumerate(
            feedback["items"],
            start=1,
        ):
            result_icon = "✅" if item["is_correct"] else "❌"
            preview_key = (
                f"quiz_source_preview_{version}_{index}"
            )

            with st.expander(
                f"{result_icon} {index}번 문제 해설",
                expanded=(
                    not item["is_correct"]
                    or st.session_state.get(preview_key, False)
                ),
            ):
                st.write(f"내 답: {item['submitted_answer']}")
                st.write(f"정답: {item['correct_answer']}")
                st.write(f"해설: {item['explanation']}")
                st.caption(
                    f"근거: {item['source_file']} p.{item['page']}"
                )
                display_quiz_source_preview(
                    item["source_file"],
                    item["page"],
                    preview_key,
                )

        if st.button(
            "새 퀴즈 만들기",
            key=f"new_student_quiz_{version}",
            use_container_width=True,
        ):
            st.session_state.student_quiz = []
            st.session_state.student_quiz_feedback = None
            sync_active_chat_session(
                "student",
                "student_chat_history",
            )
            st.rerun()


def change_source_page(state_key, step, page_count):
    """학생 출처 미리보기의 현재 페이지 위치를 이동합니다."""
    current_index = st.session_state.get(state_key, 0)
    st.session_state[state_key] = max(
        0,
        min(current_index + step, page_count - 1),
    )


def display_student_sources(sources):
    documents = load_documents()
    grouped_pages = {}

    for source in sources:
        if not isinstance(source, dict):
            continue

        file_name = Path(
            source.get(
                "file",
                st.session_state.rag_filename or ""
            )
        ).name
        metadata = documents.get(file_name)
        pdf_path = PDF_DIR / file_name

        if (
            metadata is None
            or not metadata.get("public", False)
            or not pdf_path.is_file()
        ):
            continue

        try:
            page_number = int(source.get("page"))
        except (TypeError, ValueError):
            continue

        grouped_pages.setdefault(file_name, set()).add(page_number)

    for file_index, file_name in enumerate(
        sorted(grouped_pages.keys())
    ):
        pages = sorted(grouped_pages[file_name])

        if not pages:
            continue

        pdf_path = PDF_DIR / file_name
        file_key = hashlib.sha256(
            file_name.encode("utf-8")
        ).hexdigest()[:12]
        state_key = f"source_page_index_{file_key}"
        current_index = st.session_state.get(state_key, 0)
        current_index = max(
            0,
            min(current_index, len(pages) - 1),
        )
        st.session_state[state_key] = current_index
        selected_page = pages[current_index]
        page_list = ", ".join(str(page) for page in pages)

        with st.expander(
            f"📄 {file_name} — 출처 p.{page_list}",
            expanded=(file_index == 0),
        ):
            previous_col, status_col, next_col = st.columns(
                [1, 2, 1]
            )

            with previous_col:
                st.button(
                    "◀ 이전 페이지",
                    key=f"previous_source_page_{file_key}",
                    disabled=(current_index == 0),
                    use_container_width=True,
                    on_click=change_source_page,
                    args=(state_key, -1, len(pages)),
                )

            with status_col:
                st.markdown(
                    f"<p style='text-align:center;'>"
                    f"<strong>p.{selected_page}</strong> "
                    f"({current_index + 1} / {len(pages)})"
                    f"</p>",
                    unsafe_allow_html=True,
                )

            with next_col:
                st.button(
                    "다음 페이지 ▶",
                    key=f"next_source_page_{file_key}",
                    disabled=(current_index == len(pages) - 1),
                    use_container_width=True,
                    on_click=change_source_page,
                    args=(state_key, 1, len(pages)),
                )

            try:
                modified_time_ns = pdf_path.stat().st_mtime_ns
                page_image = render_pdf_page(
                    str(pdf_path),
                    selected_page,
                    modified_time_ns
                )
                st.image(
                    page_image,
                    caption=(
                        f"{file_name} - {selected_page}페이지"
                    ),
                    width="stretch"
                )
            except Exception as e:
                log_exception("출처 페이지 미리보기 렌더링 실패", e)
                st.warning(
                    "페이지 미리보기를 표시하지 못했습니다."
                )

def display_student_downloads(file_names):
    """학생 자료 패널에 선택한 공개 PDF 다운로드 버튼을 표시합니다."""
    documents = load_documents()
    downloadable_files = []

    for file_name in file_names:
        metadata = documents.get(file_name)
        pdf_path = PDF_DIR / file_name

        if (
            metadata is not None
            and metadata.get("public", False)
            and pdf_path.is_file()
        ):
            downloadable_files.append(file_name)

    if not downloadable_files:
        st.caption("다운로드할 강의자료를 선택해주세요.")
        return

    for file_name in downloadable_files:
        pdf_path = PDF_DIR / file_name
        modified_time_ns = pdf_path.stat().st_mtime_ns
        download_key = (
            "sidebar_download_"
            + hashlib.sha256(
                file_name.encode("utf-8")
            ).hexdigest()[:12]
        )

        st.download_button(
            label=f"⬇️ {file_name}",
            data=load_pdf_bytes(
                str(pdf_path),
                modified_time_ns
            ),
            file_name=file_name,
            mime="application/pdf",
            key=download_key,
            on_click="ignore",
            width="stretch"
        )


ensure_document_storage()


TEST_USERS = {
    "professor": {
        "password_hash": (
            "a0d8bd9fa838383cc80c302972681bc5"
            "$fb432de65b608ae330d2b7a022a2ca2a1fc5455b963e821543015c7071988e13"
        ),
        "role": "professor",
        "name": "김교수"
    },
    "student": {
        "password_hash": (
            "f3ffad544834970c324e795ae18a4607"
            "$598a6e980bc815155529f2f68050a746e6e132d9accaea4faafdb3702d70bda6"
        ),
        "role": "student",
        "name": "홍길동"
    }
}


def logout():
    current_role = st.session_state.get("role")

    if current_role in ("professor", "student"):
        sync_active_chat_session(
            current_role,
            f"{current_role}_chat_history",
        )

    st.session_state.logged_in = False
    st.session_state.role = None
    st.session_state.username = None
    st.session_state.chat_context_role = None
    st.rerun()


def show_login():
    st.title("🎓 AI 학습지원 플랫폼")

    st.write(
        "강의자료를 기반으로 학습과 수업을 지원하는 "
        "RAG 기반 AI 교육 플랫폼입니다."
    )

    st.divider()

    left, center, right = st.columns([1, 2, 1])

    with center:
        st.subheader("로그인")

        with st.form(
            "login_form",
            clear_on_submit=False,
            enter_to_submit=True
        ):
            user_id = st.text_input(
                "아이디 또는 이메일",
                placeholder="아이디를 입력하세요",
                key="login_user_id",
                autocomplete="username"
            )

            password = st.text_input(
                "비밀번호",
                type="password",
                placeholder="비밀번호를 입력하세요",
                key="login_password",
                autocomplete="current-password"
            )

            submitted = st.form_submit_button(
                "로그인",
                use_container_width=True,
                type="primary"
            )

            if submitted:
                clean_id = (user_id or "").strip()
                clean_password = (password or "").strip()

                locked, remaining_seconds = security.check_lockout(
                    clean_id
                )

                if locked:
                    minutes = max(1, remaining_seconds // 60)
                    log_event(
                        f"잠금 상태 계정 로그인 시도: id={clean_id}",
                        level="warning",
                    )
                    st.error(
                        "로그인 시도 횟수를 초과했습니다. "
                        f"약 {minutes}분 후 다시 시도해주세요."
                    )
                else:
                    user = TEST_USERS.get(clean_id)

                    if (
                        user is not None
                        and security.verify_password(
                            clean_password,
                            user["password_hash"],
                        )
                    ):
                        security.register_successful_login(clean_id)
                        log_event(f"로그인 성공: id={clean_id}")
                        st.session_state.logged_in = True
                        st.session_state.role = user["role"]
                        st.session_state.username = user["name"]
                        st.rerun()
                    else:
                        security.register_failed_login(clean_id)
                        log_event(
                            f"로그인 실패: id={clean_id}",
                            level="warning",
                        )
                        st.error(
                            "아이디 또는 비밀번호를 확인해주세요."
                        )

        st.info(
            "로그인 후 시스템이 계정 역할을 확인하여 "
            "교수자 또는 학생 화면으로 이동합니다."
        )


def show_professor_page():
    ensure_chat_sessions("professor", "professor_chat_history")

    col1, col2 = st.columns([8, 2])

    with col1:
        st.title("AI 학습지원 서비스 (교수용)")

    with col2:
        st.write(
            f"👨‍🏫 {st.session_state.username} / 교수자"
        )

        if st.button(
            "로그아웃",
            use_container_width=True
        ):
            logout()

    panel_label = (
        "◀"
        if st.session_state.professor_panel_visible
        else "▶"
    )
    panel_help = (
        "자료 패널 숨기기"
        if st.session_state.professor_panel_visible
        else "자료 패널 열기"
    )

    if st.button(
        panel_label,
        key="professor_panel_toggle",
        help=panel_help,
    ):
        st.session_state.professor_panel_visible = not (
            st.session_state.professor_panel_visible
        )
        st.rerun()

    st.divider()

    documents = load_documents()
    professor_documents = sorted(documents.keys())
    saved_selection = st.session_state.get(
        "professor_selected_documents",
        professor_documents[:1]
    )
    selected_documents = [
        filename for filename in saved_selection
        if filename in professor_documents
    ]

    if st.session_state.professor_panel_visible:
        sidebar, main = st.columns([1, 3])
    else:
        sidebar = None
        main = st.container()

    if sidebar is not None:
        with sidebar:
            display_chat_session_sidebar(
                "professor",
                "professor_chat_history",
            )
            st.divider()
            st.subheader("📚 강의자료 관리")

            document_action_message = st.session_state.pop(
                "document_action_message",
                None,
            )
            if document_action_message:
                st.success(document_action_message)

            uploaded_files = st.file_uploader(
                "PDF 강의자료 업로드",
                type=["pdf"],
                accept_multiple_files=True
            )

            if st.button(
                "선택한 PDF 저장",
                use_container_width=True,
                disabled=not uploaded_files
            ):
                try:
                    with st.spinner("PDF 파일을 저장하는 중입니다..."):
                        saved_files, rejected_files = (
                            save_uploaded_pdfs(uploaded_files)
                        )

                    if saved_files:
                        st.success(
                            f"PDF {len(saved_files)}개를 저장했습니다."
                        )
                        log_event(
                            "PDF 저장 완료: "
                            + ", ".join(saved_files)
                        )

                    if rejected_files:
                        for reason in rejected_files:
                            st.warning(reason)

                    if not saved_files and not rejected_files:
                        st.warning("저장할 PDF가 없습니다.")
                except Exception as e:
                    log_exception("PDF 저장 실패", e)
                    st.error(
                        "PDF 저장 중 오류가 발생했습니다. "
                        "잠시 후 다시 시도해주세요."
                    )

            st.divider()
            st.write("**업로드된 강의자료**")

            documents = load_documents()
            professor_documents = sorted(documents.keys())

            st.write("**강의자료 삭제**")

            pending_delete = st.session_state.pending_document_delete

            if pending_delete not in professor_documents:
                st.session_state.pending_document_delete = None
                pending_delete = None

            if professor_documents and pending_delete:
                st.warning(
                    f"'{pending_delete}'을(를) 정말 삭제할까요? "
                    "삭제한 파일은 복구할 수 없습니다."
                )
                confirm_col, cancel_col = st.columns(2)

                with confirm_col:
                    confirm_delete = st.button(
                        "삭제 확인",
                        type="primary",
                        use_container_width=True,
                        key="confirm_document_delete",
                    )

                with cancel_col:
                    cancel_delete = st.button(
                        "취소",
                        use_container_width=True,
                        key="cancel_document_delete",
                    )

                if confirm_delete:
                    try:
                        if delete_document(pending_delete):
                            log_event(f"강의자료 삭제: {pending_delete}")
                            st.session_state.document_action_message = (
                                f"'{pending_delete}'을(를) 삭제했습니다."
                            )
                            st.rerun()
                        else:
                            st.warning("이미 삭제되었거나 찾을 수 없는 자료입니다.")
                    except OSError as error:
                        log_exception("PDF 삭제 실패", error)
                        st.error("PDF 삭제 중 오류가 발생했습니다.")

                if cancel_delete:
                    st.session_state.pending_document_delete = None
                    st.rerun()
            elif professor_documents:
                delete_target = st.selectbox(
                    "삭제할 강의자료 선택",
                    professor_documents,
                    key="delete_document_target",
                )

                if st.button(
                    "선택한 강의자료 삭제",
                    use_container_width=True,
                    key="request_document_delete",
                ):
                    st.session_state.pending_document_delete = delete_target
                    st.rerun()
            else:
                st.caption("삭제할 강의자료가 없습니다.")

            st.divider()

            if professor_documents:
                selected_documents = st.multiselect(
                    "AI가 사용할 자료 선택 (복수 선택 가능)",
                    professor_documents,
                    default=professor_documents[:1],
                    key="professor_selected_documents",
                    placeholder="강의자료를 하나 이상 선택하세요"
                )
            else:
                selected_documents = []
                st.info("아직 업로드된 강의자료가 없습니다.")

            st.divider()
            st.write("**학생 공개 설정**")

            if professor_documents:
                for filename in professor_documents:
                    current_public = bool(
                        documents[filename].get("public", False)
                    )
                    toggle_key = (
                        "public_"
                        + hashlib.sha256(
                            filename.encode("utf-8")
                        ).hexdigest()[:12]
                    )

                    new_public = st.toggle(
                        filename,
                        value=current_public,
                        key=toggle_key
                    )

                    if new_public != current_public:
                        set_document_public(
                            filename,
                            new_public
                        )
            else:
                st.caption("강의자료를 먼저 업로드해주세요.")

    with main:
        st.subheader("🤖 AI 학습지원 챗봇")

        if selected_documents:
            st.caption(
                "현재 선택된 강의자료: "
                + ", ".join(selected_documents)
            )

            if (
                st.session_state.rag_selection
                != tuple(sorted(selected_documents))
                or st.session_state.rag_chain is None
            ):
                try:
                    with st.spinner(
                        "선택한 PDF들을 분석하고 AI 검색 데이터를 "
                        "만드는 중입니다..."
                    ):
                        activate_documents(selected_documents)
                        st.session_state.student_quiz = []
                        st.session_state.student_quiz_feedback = None
                except Exception as e:
                    st.session_state.rag_chain = None
                    log_exception(
                        "교수 PDF 분석/임베딩 실패: "
                        + ", ".join(selected_documents),
                        e,
                    )
                    st.error(
                        "PDF 분석 중 오류가 발생했습니다. "
                        "잠시 후 다시 시도해주세요."
                    )
        else:
            st.caption("현재 선택된 강의자료가 없습니다.")

        question = st.text_area(
            "질문 입력",
            placeholder="궁금한 내용을 입력하세요...",
            height=120
        )

        col1, col2 = st.columns(2)

        with col1:
            ask_button = st.button(
                "질문하기",
                use_container_width=True,
                type="primary"
            )

        with col2:
            summary_button = st.button(
                "강의자료 요약",
                use_container_width=True
            )

        st.divider()

        if ask_button:
            if not selected_documents:
                st.warning("먼저 강의자료를 저장하고 선택해주세요.")
            elif not question.strip():
                st.warning("질문을 입력해주세요.")
            elif st.session_state.rag_chain is None:
                st.warning("먼저 PDF 강의자료를 업로드해주세요.")
            else:
                allowed, remaining = security.try_consume_call(
                    chat_owner_key("professor")
                )

                if not allowed:
                    st.warning(
                        "오늘의 AI 호출 한도를 모두 사용했습니다. "
                        "내일 다시 시도해주세요."
                    )
                else:
                    run_streaming_answer(
                        "professor_chat_history",
                        display_user_message=question,
                        question_for_chain=question,
                        spinner_text=(
                            "AI가 강의자료에서 답변을 찾는 중이에요..."
                        ),
                    )

        elif summary_button:
            if not selected_documents:
                st.warning("먼저 강의자료를 저장하고 선택해주세요.")
            elif st.session_state.rag_chain is None:
                st.warning("먼저 PDF 강의자료를 업로드해주세요.")
            else:
                allowed, remaining = security.try_consume_call(
                    chat_owner_key("professor")
                )

                if not allowed:
                    st.warning(
                        "오늘의 AI 호출 한도를 모두 사용했습니다. "
                        "내일 다시 시도해주세요."
                    )
                else:
                    try:
                        with st.spinner(
                            "AI가 선택한 강의자료를 요약하는 중이에요...",
                            show_time=True,
                        ):
                            result = (
                                st.session_state.rag_chain.generate_summary()
                            )
                            st.session_state.rag_result = result
                            append_chat_history(
                                "professor_chat_history",
                                "선택한 강의자료를 요약해줘",
                                result,
                            )
                    except Exception as e:
                        log_exception("교수 강의자료 요약 실패", e)
                        st.error(
                            "강의자료 요약 중 오류가 발생했습니다. "
                            "잠시 후 다시 시도해주세요."
                        )

        display_chat_history(
            "professor_chat_history",
            "clear_professor_chat_history",
        )

        st.subheader("📖 출처 / 근거")

        if st.session_state.rag_result:
            result = st.session_state.rag_result

            if isinstance(result, dict):
                sources = result.get("sources", [])

                if sources:
                    display_grouped_sources(sources)
                else:
                    st.write(
                        "표시할 출처가 없습니다."
                    )
            else:
                st.write(
                    "출처 정보를 확인할 수 없습니다."
                )
        else:
            st.write(
                "질문에 답변하면 근거가 여기에 표시됩니다."
            )


def show_student_page():
    ensure_chat_sessions("student", "student_chat_history")

    col1, col2 = st.columns([8, 2])

    with col1:
        st.title("AI 학습지원 서비스 (학생용)")

    with col2:
        st.write(
            f"🎓 {st.session_state.username} / 학생"
        )

        if st.button(
            "로그아웃",
            use_container_width=True
        ):
            logout()

    panel_label = (
        "◀"
        if st.session_state.student_panel_visible
        else "▶"
    )
    panel_help = (
        "자료 패널 숨기기"
        if st.session_state.student_panel_visible
        else "자료 패널 열기"
    )

    if st.button(
        panel_label,
        key="student_panel_toggle",
        help=panel_help,
    ):
        st.session_state.student_panel_visible = not (
            st.session_state.student_panel_visible
        )
        st.rerun()

    st.divider()

    documents = load_documents()
    allowed_documents = sorted(
        filename
        for filename, metadata in documents.items()
        if metadata.get("public", False)
    )
    saved_selection = st.session_state.get(
        "student_selected_documents",
        allowed_documents[:1]
    )
    selected_documents = [
        filename for filename in saved_selection
        if filename in allowed_documents
    ]

    if st.session_state.student_panel_visible:
        sidebar, main = st.columns([1, 3])
    else:
        sidebar = None
        main = st.container()

    if sidebar is not None:
        with sidebar:
            display_chat_session_sidebar(
                "student",
                "student_chat_history",
            )
            st.divider()
            st.subheader("📚 강의자료")

            st.caption(
                "접근 가능한 강의자료만 표시됩니다."
            )

            documents = load_documents()
            allowed_documents = sorted(
                filename
                for filename, metadata in documents.items()
                if metadata.get("public", False)
            )

            if allowed_documents:
                selected_documents = st.multiselect(
                    "강의자료 선택 (복수 선택 가능)",
                    allowed_documents,
                    default=allowed_documents[:1],
                    key="student_selected_documents",
                    placeholder="공개 강의자료를 하나 이상 선택하세요"
                )
            else:
                selected_documents = []
                st.info("현재 공개된 강의자료가 없습니다.")

            st.divider()
            st.write("**선택한 자료 다운로드**")
            display_student_downloads(selected_documents)

    with main:
        st.subheader("🤖 AI 학습 도우미")

        answer_language = st.selectbox(
            "답변 언어",
            list(STUDENT_ANSWER_LANGUAGES),
            key="student_answer_language",
            help="질문하기와 쉽게 설명해줘의 답변 언어를 선택합니다.",
        )
        answer_instruction = STUDENT_ANSWER_LANGUAGES[answer_language]

        if selected_documents:
            st.caption(
                "현재 선택된 강의자료: "
                + ", ".join(selected_documents)
            )

            if (
                st.session_state.rag_selection
                != tuple(sorted(selected_documents))
                or st.session_state.rag_chain is None
            ):
                try:
                    with st.spinner(
                        "선택한 강의자료들을 준비하는 중입니다..."
                    ):
                        activate_documents(selected_documents)
                except Exception as e:
                    st.session_state.rag_chain = None
                    log_exception(
                        "학생 PDF 분석/임베딩 실패: "
                        + ", ".join(selected_documents),
                        e,
                    )
                    st.error(
                        "PDF 분석 중 오류가 발생했습니다. "
                        "잠시 후 다시 시도해주세요."
                    )
        else:
            st.caption("현재 선택된 강의자료가 없습니다.")

        question = st.text_area(
            "질문 입력",
            placeholder="궁금한 내용을 입력하세요...",
            height=120,
            key="student_question"
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            ask_button = st.button(
                "질문하기",
                use_container_width=True,
                type="primary",
                key="student_ask"
            )

        with col2:
            explain_button = st.button(
                "쉽게 설명해줘 / 맞춤형 설명",
                use_container_width=True,
                key="student_explain"
            )

        with col3:
            quiz_button = st.button(
                "퀴즈 풀기",
                use_container_width=True,
                key="student_quiz_button"
            )

        st.divider()
        if ask_button:
            if not selected_documents:
                st.warning("현재 공개된 강의자료가 없습니다.")
            elif not question.strip():
                st.warning("질문을 입력해주세요.")
            elif st.session_state.rag_chain is None:
                st.warning("먼저 PDF 강의자료를 업로드해주세요.")
            else:
                allowed, remaining = security.try_consume_call(
                    chat_owner_key("student")
                )

                if not allowed:
                    st.warning(
                        "오늘의 AI 호출 한도를 모두 사용했습니다. "
                        "내일 다시 시도해주세요."
                    )
                else:
                    run_streaming_answer(
                        "student_chat_history",
                        display_user_message=question,
                        question_for_chain=question,
                        spinner_text=(
                            "AI가 강의자료에서 답변을 찾는 중이에요..."
                        ),
                        answer_instruction=answer_instruction,
                    )

        elif explain_button:
            if not selected_documents:
                st.warning("현재 공개된 강의자료가 없습니다.")
            elif not question.strip():
                st.warning("설명받고 싶은 내용을 입력해주세요.")
            elif st.session_state.rag_chain is None:
                st.warning("먼저 PDF 강의자료를 업로드해주세요.")
            else:
                allowed, remaining = security.try_consume_call(
                    chat_owner_key("student")
                )

                if not allowed:
                    st.warning(
                        "오늘의 AI 호출 한도를 모두 사용했습니다. "
                        "내일 다시 시도해주세요."
                    )
                else:
                    run_streaming_answer(
                        "student_chat_history",
                        display_user_message=f"더 쉽게 설명해줘: {question}",
                        question_for_chain=(
                            "다음 질문에 대해 학생이 이해하기 쉽도록 "
                            "쉬운 표현과 구체적인 예시를 사용해서 설명해주세요.\n\n"
                            f"질문: {question}"
                        ),
                        spinner_text=(
                            "AI가 내용을 더 쉽게 정리하는 중이에요..."
                        ),
                        answer_instruction=answer_instruction,
                    )

        elif quiz_button:
            if not selected_documents:
                st.warning("현재 공개된 강의자료가 없습니다.")
            elif st.session_state.rag_chain is None:
                st.warning("먼저 강의자료를 선택해주세요.")
            else:
                allowed, remaining = security.try_consume_call(
                    chat_owner_key("student")
                )

                if not allowed:
                    st.warning(
                        "오늘의 AI 호출 한도를 모두 사용했습니다. "
                        "내일 다시 시도해주세요."
                    )
                else:
                    try:
                        with st.spinner(
                            "AI가 풀 수 있는 퀴즈를 만드는 중이에요...",
                            show_time=True,
                        ):
                            result = (
                                st.session_state.rag_chain.generate_quiz()
                            )
                            quiz = result.get("quiz", [])

                            if not quiz:
                                st.warning(
                                    "퀴즈 문항을 만들지 못했습니다. "
                                    "다시 시도해주세요."
                                )
                            else:
                                st.session_state.student_quiz = quiz
                                st.session_state.student_quiz_feedback = None
                                st.session_state.student_quiz_version += 1
                                st.session_state.rag_result = {
                                    "answer": "복습 퀴즈 6문항을 만들었습니다.",
                                    "sources": result.get("sources", []),
                                    "has_evidence": True,
                                }
                                sync_active_chat_session(
                                    "student",
                                    "student_chat_history",
                                )
                    except Exception as e:
                        log_exception("학생 퀴즈 생성 실패", e)
                        st.error(
                            "퀴즈 생성 중 오류가 발생했습니다. "
                            "잠시 후 다시 시도해주세요."
                        )

        display_student_quiz()

        display_chat_history(
            "student_chat_history",
            "clear_student_chat_history",
        )

        st.subheader("📖 출처 / 근거")

        if st.session_state.rag_result:
            result = st.session_state.rag_result

            if isinstance(result, dict):
                sources = result.get("sources", [])

                if sources:
                    display_grouped_sources(sources)
                    display_student_source_excerpts(sources)
                else:
                    st.write(
                        "표시할 출처가 없습니다."
                    )
            else:
                st.write(
                    "출처 정보를 확인할 수 없습니다."
                )
        else:
            st.write(
                "질문에 답변하면 근거가 여기에 표시됩니다."
            )


if not st.session_state.logged_in:
    show_login()
elif st.session_state.role == "professor":
    show_professor_page()
elif st.session_state.role == "student":
    show_student_page()
else:
    st.error(
        "사용자 역할 정보를 확인할 수 없습니다."
    )

    if st.button("로그아웃"):
        logout()
