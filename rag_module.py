import hashlib
import logging
import os
import re
import time
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
import pymupdf as fitz
from pydantic import BaseModel, Field

from langchain_community.vectorstores import FAISS
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

logger = logging.getLogger("ai_learning_platform")

CHAT_MODEL = "gemini-3.6-flash"
EMBEDDING_MODEL = "models/gemini-embedding-001"

RELEVANCE_THRESHOLD = 0.55


def get_embeddings():
    """Gemini 임베딩 객체를 생성합니다. (GOOGLE_API_KEY 환경변수 사용)"""
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)


def content_to_text(content) -> str:
    """
    LLM 응답의 content를 사람이 읽을 깔끔한 문자열로 변환합니다.

    OpenAI는 content가 항상 문자열이지만, Gemini(langchain-google-genai)는
    경우에 따라 리스트(예: [{'type': 'text', 'text': '...'}]) 형태로 돌려줍니다.
    이 함수는 문자열/리스트/딕셔너리 어떤 형태가 와도 텍스트만 이어붙여 반환해,
    화면에 '[{...}]' 같은 원본 구조가 그대로 노출되는 문제를 방지합니다.
    """
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            else:
                text_attr = getattr(item, "text", None)
                if isinstance(text_attr, str):
                    parts.append(text_attr)
        return "".join(parts)

    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text

    return str(content)

EMBEDDING_BATCH_SIZE = 20
EMBEDDING_BATCH_DELAY = 15
EMBEDDING_MAX_RETRIES = 3
EMBEDDING_RETRY_BASE_DELAY = 15

CHUNK_SIZE = 700
CHUNK_OVERLAP = 120

CHUNK_SEPARATORS = [
    "\n\n",
    "\n",
    ". ",
    "? ",
    "! ",
    "다. ",
    "요. ",
    "。",
    " ",
    "",
]


class QuizQuestion(BaseModel):
    question_type: Literal["ox", "short_answer"]
    question: str
    answer: str
    accepted_answers: list[str] = Field(default_factory=list)
    explanation: str
    source_file: str
    page: int = Field(ge=1)


class QuizSet(BaseModel):
    questions: list[QuizQuestion] = Field(
        min_length=6,
        max_length=6,
    )


def load_pdf_documents(pdf_path: str):
    """
    PDF의 각 페이지를 Document로 변환합니다.
    metadata에 파일명과 실제 PDF 페이지 번호를 저장합니다.
    """
    pdf = fitz.open(pdf_path)
    documents = []

    filename = Path(pdf_path).name

    try:
        for page_index, page in enumerate(pdf):
            text = page.get_text("text").strip()

            if not text:
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": filename,
                        "page": page_index + 1,
                    },
                )
            )
    finally:
        pdf.close()

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS,
        keep_separator=True,
    )

    chunks = splitter.split_documents(documents)

    return [
        chunk
        for chunk in chunks
        if chunk.page_content and chunk.page_content.strip()
    ]


def _call_with_retry(func, *args, **kwargs):
    """
    임베딩 API 호출용 재시도 헬퍼.
    rate limit, 일시적 네트워크 오류 등에 대비해
    지수 백오프(15초 → 30초 → 60초)로 최대 3회 재시도합니다.
    """
    last_error = None

    for attempt in range(1, EMBEDDING_MAX_RETRIES + 1):
        try:
            return func(*args, **kwargs)
        except Exception as error:
            last_error = error

            if attempt >= EMBEDDING_MAX_RETRIES:
                break

            delay = EMBEDDING_RETRY_BASE_DELAY * (2 ** (attempt - 1))
            logger.warning(
                "임베딩 API 호출 실패 (시도 %d/%d), %d초 후 재시도: %s",
                attempt,
                EMBEDDING_MAX_RETRIES,
                delay,
                error,
            )
            time.sleep(delay)

    logger.error(
        "임베딩 API 호출 최종 실패 (%d회 시도): %s",
        EMBEDDING_MAX_RETRIES,
        last_error,
    )
    raise RuntimeError(
        f"임베딩 API 호출에 {EMBEDDING_MAX_RETRIES}회 실패했습니다: "
        f"{last_error}"
    ) from last_error


def build_vectorstore(chunks, progress_callback=None):
    """
    청크를 배치 단위로 나누어 임베딩합니다.

    - 배치마다 재시도 로직을 적용해 일시적 API 오류에 견고합니다.
    - progress_callback(done, total)을 배치가 끝날 때마다 호출해서
      대용량 PDF 처리 중에도 진행 상황을 화면에 표시할 수 있습니다.
    """
    if not chunks:
        raise ValueError(
            "PDF에서 읽을 수 있는 텍스트를 찾지 못했습니다."
        )

    embeddings = get_embeddings()

    total = len(chunks)
    vectorstore = None

    for start in range(0, total, EMBEDDING_BATCH_SIZE):
        batch = chunks[start:start + EMBEDDING_BATCH_SIZE]

        if vectorstore is None:
            vectorstore = _call_with_retry(
                FAISS.from_documents,
                batch,
                embedding=embeddings,
            )
        else:
            _call_with_retry(
                vectorstore.add_documents,
                batch,
            )

        done = min(start + EMBEDDING_BATCH_SIZE, total)

        if progress_callback is not None:
            progress_callback(done, total)

        if done < total:
            logger.info(
                "임베딩 배치 완료 (%d/%d), %d초 대기 후 다음 배치를 처리합니다.",
                done,
                total,
                EMBEDDING_BATCH_DELAY,
            )
            time.sleep(EMBEDDING_BATCH_DELAY)

    return vectorstore


def compute_content_hash(pdf_paths) -> str:
    """
    선택한 PDF들의 실제 내용을 기준으로 캐시 키를 만듭니다.
    파일명이 아니라 내용(바이트) 해시라서, 같은 이름이라도 내용이
    바뀌면 자동으로 다른 캐시 키가 되어 새로 임베딩합니다.

    청킹/임베딩 설정도 키에 포함합니다. 청크 크기 등 설정이 바뀌면
    이전 인덱스와 호환되지 않으므로, 자동으로 다른 키가 되어
    새 인덱스를 만들도록 합니다.
    """
    hasher = hashlib.sha256()

    config_signature = (
        f"cs={CHUNK_SIZE};co={CHUNK_OVERLAP};"
        f"sep={len(CHUNK_SEPARATORS)};emb={EMBEDDING_MODEL}"
    )
    hasher.update(config_signature.encode("utf-8"))

    for pdf_path in sorted(pdf_paths, key=lambda p: Path(p).name):
        file_bytes = Path(pdf_path).read_bytes()
        hasher.update(Path(pdf_path).name.encode("utf-8"))
        hasher.update(hashlib.sha256(file_bytes).digest())

    return hasher.hexdigest()[:24]


def load_cached_vectorstore(cache_dir, content_hash, embeddings):
    """디스크에 저장된 FAISS 인덱스가 있으면 불러오고, 없거나 손상되었으면 None."""
    cache_path = Path(cache_dir) / content_hash

    if not cache_path.exists():
        return None

    try:
        return FAISS.load_local(
            str(cache_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    except Exception:
        return None


def save_vectorstore_cache(vectorstore, cache_dir, content_hash):
    """FAISS 인덱스를 디스크에 저장합니다. 실패해도 앱 동작에는 영향 없음(best-effort)."""
    cache_path = Path(cache_dir) / content_hash
    cache_path.mkdir(parents=True, exist_ok=True)

    try:
        vectorstore.save_local(str(cache_path))
    except Exception:
        pass


def prune_vector_cache(cache_dir, keep_hashes=None, max_entries=20):
    """
    디스크 벡터 캐시 폴더를 정리합니다.

    - keep_hashes(집합)가 주어지면, 그 목록에 없는 폴더를 우선 삭제합니다.
      (예: 현재 존재하는 PDF들로 만들 수 있는 유효한 캐시 해시 집합)
    - 그 후에도 폴더 수가 max_entries를 넘으면, 가장 오래된(수정시각 기준)
      폴더부터 삭제해 개수를 제한합니다.

    best-effort로 동작하며, 개별 삭제 실패는 무시합니다.
    """
    import shutil

    cache_dir = Path(cache_dir)

    if not cache_dir.exists():
        return

    entries = [p for p in cache_dir.iterdir() if p.is_dir()]

    if keep_hashes is not None:
        keep = set(keep_hashes)
        for entry in list(entries):
            if entry.name not in keep:
                try:
                    shutil.rmtree(entry, ignore_errors=True)
                except OSError:
                    pass

        entries = [p for p in cache_dir.iterdir() if p.is_dir()]

    if len(entries) > max_entries:
        entries.sort(key=lambda p: p.stat().st_mtime)

        for entry in entries[:len(entries) - max_entries]:
            try:
                shutil.rmtree(entry, ignore_errors=True)
            except OSError:
                pass


PROMPT = """
# 역할
당신은 강의자료를 기반으로 답변하는 AI 학습지원 도우미입니다.

# 반드시 지켜야 할 규칙
1. 아래 [강의자료 근거]에 포함된 정보만 사용하세요.
2. 근거에 없는 사실을 추측하거나 만들어내지 마세요.
3. 질문에 답할 근거가 충분하지 않다면 반드시
   "선택한 강의자료에서 충분한 근거를 찾지 못했습니다."
   라고 답하세요.
4. 일반 상식이나 학습된 외부 지식을 추가하지 마세요.
5. 숫자, 용어, 규격, 날짜 등은 근거의 내용을 정확하게 사용하세요.
6. 답변은 한국어로 작성하세요.

# 답변 작성 방식
1. 학생이 처음 배우는 상황을 가정하고 쉬운 표현으로 설명하세요.
2. 단순한 한 문단으로 끝내지 말고, 강의자료에 근거가 있는 범위에서
   다음 순서로 답변하세요.
   - **핵심 정의**: 질문한 개념을 1~2문장으로 정의
   - **공식과 기호**: 관련 공식이 있으면 공식과 각 기호의 의미 설명
   - **물리적 의미**: 값이 커지거나 작아질 때 무엇을 의미하는지 설명
   - **쉽게 이해하기**: 강의자료에서 뒷받침되는 직관적 설명
   - **핵심 정리**: 시험 전에 기억할 내용을 2~3개 항목으로 정리
3. 공식은 LaTeX나 수식 기호($, \\frac 등) 없이, 한 줄로 읽을 수 있는 일반 텍스트로 쓰세요.
   예: "Z = I / c", "sigma = P / A", "ΔU = Q - W" 처럼 표기하세요.
   분수는 슬래시(/)로, 제곱은 h^2 또는 h²처럼 표기하세요.
4. 강의자료에 없는 예시, 수치 또는 적용 사례는 만들어내지 마세요.
5. 질문에 따라 해당되지 않는 항목은 억지로 작성하지 말고 생략하세요.
6. 답변은 충분히 설명하되 불필요하게 반복하지 마세요.

[강의자료 근거]
{context}

[질문]
{question}

[답변]
"""


QUIZ_PROMPT = """
# 역할
당신은 선택한 강의자료만을 이용해 복습 문제를 만드는 출제자입니다.

# 반드시 지켜야 할 규칙
1. 아래 [강의자료 근거]에 포함된 정보만 사용하세요.
2. 외부 지식, 임의의 수치, 근거에 없는 예시는 추가하지 마세요.
3. OX 퀴즈 3문항과 단답형 퀴즈 3문항을 정확히 만드세요.
4. 각 문항은 강의자료만 읽으면 답할 수 있어야 합니다.
5. 같은 개념을 표현만 바꾸어 반복 출제하지 마세요.
6. 여러 파일이 제공되면 가능한 한 서로 다른 파일의 내용을 고르게 사용하세요.
7. 정답, 1~2문장의 해설, 근거 파일명과 페이지를 모든 문항에 표시하세요.
8. 한국어로 작성하세요.

# 문항 데이터 규칙
1. OX 문항의 question_type은 "ox", answer는 "O" 또는 "X"로 작성하세요.
2. 단답형 문항의 question_type은 "short_answer"로 작성하세요.
3. 단답형 accepted_answers에는 정답으로 인정할 수 있는 짧은 표현을 1~3개 넣으세요.
4. source_file과 page는 아래 근거에 실제로 표시된 파일명과 페이지를 사용하세요.

[강의자료 근거]
{context}

[퀴즈]
"""


SUMMARY_PROMPT = """
# 역할
당신은 선택한 강의자료를 학생의 복습 노트 형태로 정리하는 조교입니다.

# 반드시 지켜야 할 규칙
1. 아래 [강의자료 근거]에 포함된 정보만 사용하세요.
2. 외부 지식이나 근거에 없는 내용을 추가하지 마세요.
3. 여러 파일이 제공되면 파일별로 구분하여 모두 요약하세요.
4. 핵심 용어, 공식, 기호, 조건은 근거의 표현을 정확히 사용하세요.
5. 중요한 설명마다 근거 파일명과 페이지를 표시하세요.
6. 근거에서 확인되지 않는 항목은 억지로 작성하지 말고 생략하세요.
7. 한국어로 작성하세요.
8. 공식은 LaTeX나 수식 기호($, \\frac 등) 없이 한 줄 일반 텍스트로 쓰세요.
   예: "Z = I / c", "ΔU = Q - W", "I = b × h^3 / 12" 처럼 표기하세요.

# 출력 형식
# 강의자료 핵심 요약

## 파일명
- **핵심 주제**: 이 자료가 다루는 중심 내용
- **핵심 개념**: 시험 전에 알아야 할 개념을 항목별로 정리
- **공식과 기호**: 확인되는 공식과 기호의 의미
- **시험 전 체크**: 반드시 기억할 내용을 2~3개로 정리
- **근거 페이지**: 사용한 주요 페이지

마지막에는 선택한 전체 자료를 관통하는 내용을
`## 전체 핵심 정리` 아래 3~5개 항목으로 정리하세요.

[강의자료 근거]
{context}

[요약]
"""


def get_prompt():
    return ChatPromptTemplate.from_template(PROMPT)


def get_quiz_prompt():
    return ChatPromptTemplate.from_template(QUIZ_PROMPT)


def get_summary_prompt():
    return ChatPromptTemplate.from_template(SUMMARY_PROMPT)


def extract_search_keywords(question: str):
    tokens = re.findall(
        r"[가-힣A-Za-z0-9]+",
        question.lower(),
    )

    stop_words = {
        "뭐야",
        "무엇",
        "무엇이야",
        "설명",
        "설명해줘",
        "알려줘",
        "정의",
        "요약",
        "요약해줘",
        "정리",
        "정리해줘",
        "내용",
        "전체",
        "전부",
        "모든",
        "자료",
        "강의자료",
        "pdf",
        "문서",
        "파일",
        "선택한",
        "핵심",
        "전반적",
        "전반적인",
    }
    particles = (
        "에서",
        "으로",
        "이란",
        "란",
        "은",
        "는",
        "이",
        "가",
        "을",
        "를",
        "의",
        "에",
    )

    keywords = []

    for token in tokens:
        if token in stop_words:
            continue

        cleaned = token

        for particle in particles:
            if (
                cleaned.endswith(particle)
                and len(cleaned) > len(particle) + 1
            ):
                cleaned = cleaned[:-len(particle)]
                break

        if len(cleaned) >= 2 and cleaned not in stop_words:
            keywords.append(cleaned)

    return list(dict.fromkeys(keywords))


def is_broad_summary_request(question: str):
    """선택 자료 전체를 대상으로 하는 포괄적 요약 요청인지 확인합니다."""
    normalized_question = " ".join(str(question or "").lower().split())
    summary_signals = (
        "요약",
        "정리",
        "핵심 내용",
        "핵심내용",
    )

    if not any(
        signal in normalized_question
        for signal in summary_signals
    ):
        return False

    return not extract_search_keywords(normalized_question)


class RAGChain:
    def __init__(self, vectorstore, filename, chunks):
        self.vectorstore = vectorstore
        self.filename = filename
        self.chunks = chunks

        self.llm = ChatGoogleGenerativeAI(
            model=CHAT_MODEL,
        )

        self.prompt = get_prompt()
        self.quiz_prompt = get_quiz_prompt()
        self.summary_prompt = get_summary_prompt()
        self.quiz_llm = self.llm.with_structured_output(QuizSet)

    def generate_summary(self):
        """선택한 강의자료를 파일별 복습 노트 형태로 요약합니다."""
        source_names = list(dict.fromkeys(
            doc.metadata.get("source", self.filename)
            for doc in self.chunks
        ))

        summary_documents = []
        search_query = "목차 학습 목표 핵심 개념 정의 공식 요약 결론"

        for source_name in source_names:
            source_chunks = [
                doc for doc in self.chunks
                if doc.metadata.get("source") == source_name
            ]
            documents = self.vectorstore.similarity_search(
                search_query,
                k=4,
                filter={"source": source_name},
            )

            if source_chunks:
                positions = {
                    0,
                    len(source_chunks) // 3,
                    (len(source_chunks) * 2) // 3,
                    len(source_chunks) - 1,
                }
                documents.extend(
                    source_chunks[index]
                    for index in sorted(positions)
                )

            unique_for_source = []
            seen_for_source = set()

            for doc in documents:
                key = (
                    doc.metadata.get("page"),
                    doc.page_content,
                )
                if key not in seen_for_source:
                    seen_for_source.add(key)
                    unique_for_source.append(doc)

                if len(unique_for_source) >= 6:
                    break

            summary_documents.extend(unique_for_source)

        if not summary_documents:
            return {
                "answer": "요약할 수 있는 강의자료 내용을 찾지 못했습니다.",
                "sources": [],
                "has_evidence": False,
            }

        context_parts = []

        for doc in summary_documents:
            source = doc.metadata.get("source", self.filename)
            page = doc.metadata.get("page", "?")
            context_parts.append(
                f"[출처: {source}, 페이지: {page}]\n"
                f"{doc.page_content}"
            )

        messages = self.summary_prompt.format_messages(
            context="\n\n".join(context_parts),
        )
        response = self.llm.invoke(messages)

        sources = []
        seen_sources = set()

        for doc in summary_documents:
            source = doc.metadata.get("source", self.filename)
            page = doc.metadata.get("page", "?")
            key = (source, page)

            if key not in seen_sources:
                seen_sources.add(key)
                sources.append({
                    "file": source,
                    "page": page,
                    "excerpt": doc.page_content.strip(),
                })

        return {
            "answer": content_to_text(response.content),
            "sources": sources,
            "has_evidence": True,
        }

    def generate_quiz(self):
        """선택한 강의자료에서 OX 3문항과 단답형 3문항을 만듭니다."""
        source_names = list(dict.fromkeys(
            doc.metadata.get("source", self.filename)
            for doc in self.chunks
        ))

        quiz_documents = []
        search_query = "핵심 개념 정의 원리 공식 특징 학습 목표"

        for source_name in source_names:
            documents = self.vectorstore.similarity_search(
                search_query,
                k=3,
                filter={"source": source_name},
            )

            if not documents:
                source_chunks = [
                    doc for doc in self.chunks
                    if doc.metadata.get("source") == source_name
                ]

                if source_chunks:
                    positions = {
                        0,
                        len(source_chunks) // 2,
                        len(source_chunks) - 1,
                    }
                    documents = [
                        source_chunks[index]
                        for index in sorted(positions)
                    ]

            quiz_documents.extend(documents)

        unique_documents = []
        seen_chunks = set()

        for doc in quiz_documents:
            key = (
                doc.metadata.get("source"),
                doc.metadata.get("page"),
                doc.page_content,
            )
            if key not in seen_chunks:
                seen_chunks.add(key)
                unique_documents.append(doc)

        if not unique_documents:
            return {
                "answer": "퀴즈를 만들 수 있는 강의자료 내용을 찾지 못했습니다.",
                "sources": [],
                "has_evidence": False,
            }

        context_parts = []

        for doc in unique_documents:
            source = doc.metadata.get("source", self.filename)
            page = doc.metadata.get("page", "?")
            context_parts.append(
                f"[출처: {source}, 페이지: {page}]\n"
                f"{doc.page_content}"
            )

        messages = self.quiz_prompt.format_messages(
            context="\n\n".join(context_parts),
        )
        quiz_set = self.quiz_llm.invoke(messages)
        quiz_items = [
            question.model_dump()
            for question in quiz_set.questions
        ]

        ox_questions = [
            item for item in quiz_items
            if item["question_type"] == "ox"
        ]
        short_questions = [
            item for item in quiz_items
            if item["question_type"] == "short_answer"
        ]

        if len(ox_questions) != 3 or len(short_questions) != 3:
            raise ValueError(
                "OX 3문항과 단답형 3문항을 생성하지 못했습니다. "
                "다시 시도해주세요."
            )

        answer_parts = ["## OX 퀴즈"]

        for index, item in enumerate(ox_questions, start=1):
            answer_parts.extend([
                f"{index}. {item['question']}",
                f"   - 정답: {item['answer']}",
                f"   - 해설: {item['explanation']}",
                f"   - 근거: {item['source_file']} p.{item['page']}",
            ])

        answer_parts.append("\n## 단답형 퀴즈")

        for index, item in enumerate(short_questions, start=1):
            answer_parts.extend([
                f"{index}. {item['question']}",
                f"   - 정답: {item['answer']}",
                f"   - 해설: {item['explanation']}",
                f"   - 근거: {item['source_file']} p.{item['page']}",
            ])

        sources = []
        seen_sources = set()

        for doc in unique_documents:
            source = doc.metadata.get("source", self.filename)
            page = doc.metadata.get("page", "?")
            key = (source, page)

            if key not in seen_sources:
                seen_sources.add(key)
                sources.append({
                    "file": source,
                    "page": page,
                })

        return {
            "answer": "\n".join(answer_parts),
            "quiz": quiz_items,
            "sources": sources,
            "has_evidence": True,
        }

    def _retrieve_context(self, question: str):
        """
        질문에 대한 근거 청크를 검색해 Context 문자열과 출처 목록을 만듭니다.
        invoke()와 stream()이 공유하는 검색 로직입니다.

        반환:
        - relevant_results가 비어 있으면 (None, None) — 근거 부족
        - 그렇지 않으면 (context 문자열, sources 리스트)
        """
        results = (
            self.vectorstore
            .similarity_search_with_relevance_scores(
                question,
                k=8,
            )
        )

        threshold = RELEVANCE_THRESHOLD

        relevant_results = [
            (doc, score)
            for doc, score in results
            if score >= threshold
        ]

        keywords = extract_search_keywords(question)

        if keywords:
            existing_chunks = {
                (
                    doc.metadata.get("source"),
                    doc.metadata.get("page"),
                    doc.page_content,
                )
                for doc, score in relevant_results
            }

            for doc in self.chunks:
                content = doc.page_content.lower()

                if not any(keyword in content for keyword in keywords):
                    continue

                chunk_key = (
                    doc.metadata.get("source"),
                    doc.metadata.get("page"),
                    doc.page_content,
                )

                if chunk_key not in existing_chunks:
                    relevant_results.append((doc, 1.0))
                    existing_chunks.add(chunk_key)

                if len(relevant_results) >= 8:
                    break

        if not relevant_results:
            return None, None

        context_parts = []

        for doc, score in relevant_results:
            source = doc.metadata.get("source", self.filename)
            page = doc.metadata.get("page", "?")
            context_parts.append(
                f"[출처: {source}, 페이지: {page}]\n"
                f"{doc.page_content}"
            )

        context = "\n\n".join(context_parts)

        sources = []
        seen = set()

        for doc, score in relevant_results:
            source = doc.metadata.get("source", self.filename)
            page = doc.metadata.get("page", "?")
            key = (source, page)

            if key not in seen:
                seen.add(key)
                sources.append({
                    "file": source,
                    "page": page,
                    "score": round(float(score), 3),
                    "excerpt": doc.page_content.strip(),
                })

        return context, sources

    def _build_prompt_messages(self, question, context, answer_instruction=""):
        """LLM에 보낼 메시지를 구성합니다."""
        prompt_question = question

        if answer_instruction.strip():
            prompt_question = (
                f"{question}\n\n"
                "[답변 작성 언어 지침]\n"
                f"{answer_instruction.strip()}"
            )

        return self.prompt.format_messages(
            context=context,
            question=prompt_question,
        )

    def stream(self, question: str, answer_instruction: str = ""):
        """
        답변을 토큰 단위로 스트리밍하는 제너레이터를 돌려줍니다.

        사용법 (app.py):
            stream_gen, meta = chain.stream(question)
            answer_text = st.write_stream(stream_gen)   # 실시간 출력
            # 이후 meta["sources"], meta["has_evidence"]로 마무리 처리

        반환: (generator, meta_dict)
        - meta_dict["has_evidence"]가 False면 generator는 근거 부족 메시지를
          한 번에 내보내고, sources는 빈 리스트입니다.
        - LLM 스트리밍 중 오류가 나면 generator가 예외를 발생시키므로
          호출부에서 try/except로 감싸야 합니다.
        """
        if is_broad_summary_request(question):
            result = self.generate_summary()

            def _summary_gen():
                yield result.get("answer", "")

            return _summary_gen(), {
                "sources": result.get("sources", []),
                "has_evidence": result.get("has_evidence", True),
            }

        context, sources = self._retrieve_context(question)

        if context is None:
            no_evidence_message = (
                "선택한 강의자료에서 충분한 근거를 찾지 못했습니다."
            )

            def _no_evidence_gen():
                yield no_evidence_message

            return _no_evidence_gen(), {
                "sources": [],
                "has_evidence": False,
            }

        messages = self._build_prompt_messages(
            question,
            context,
            answer_instruction,
        )

        def _token_gen():
            for chunk in self.llm.stream(messages):
                text = content_to_text(getattr(chunk, "content", ""))
                if text:
                    yield text

        return _token_gen(), {
            "sources": sources,
            "has_evidence": True,
        }

    def invoke(
        self,
        question: str,
        answer_instruction: str = "",
    ):
        """
        기존 app.py에서 chain.invoke(question) 형태로
        사용할 수 있도록 만든 함수입니다.

        반환:
        {
            "answer": "...",
            "sources": [...],
            "has_evidence": True/False
        }
        """

        if is_broad_summary_request(question):
            return self.generate_summary()

        context, sources = self._retrieve_context(question)

        if context is None:
            return {
                "answer": (
                    "선택한 강의자료에서 충분한 "
                    "근거를 찾지 못했습니다."
                ),
                "sources": [],
                "has_evidence": False,
            }

        messages = self._build_prompt_messages(
            question,
            context,
            answer_instruction,
        )

        response = self.llm.invoke(messages)
        answer = content_to_text(response.content)

        return {
            "answer": answer,
            "sources": sources,
            "has_evidence": True,
        }


def process_pdfs_and_get_chain(
    pdf_paths,
    cache_dir=None,
    progress_callback=None,
):
    """
    PDF들을 읽어 RAGChain을 만듭니다.

    cache_dir가 주어지면, PDF 내용 해시로 디스크에 저장된 FAISS 인덱스를
    먼저 찾아봅니다. 캐시가 있으면 임베딩 API 호출 없이 즉시 재사용하고,
    없으면 새로 임베딩한 뒤 다음을 위해 디스크에 저장합니다.
    (PDF 텍스트 추출·청크 분할은 API 호출이 아니라서 캐시 여부와 무관하게
    항상 다시 수행합니다 — RAGChain이 원문 청크를 직접 참조하기 때문입니다.)

    progress_callback(done, total)은 임베딩이 실제로 실행될 때만 호출됩니다.
    """
    documents = []
    filenames = []

    for pdf_path in pdf_paths:
        documents.extend(
            load_pdf_documents(pdf_path)
        )
        filenames.append(Path(pdf_path).name)

    chunks = split_documents(documents)

    vectorstore = None

    if cache_dir is not None:
        content_hash = compute_content_hash(pdf_paths)
        embeddings = get_embeddings()
        vectorstore = load_cached_vectorstore(
            cache_dir,
            content_hash,
            embeddings,
        )

        if vectorstore is not None and progress_callback is not None:
            progress_callback(len(chunks), len(chunks))

    if vectorstore is None:
        vectorstore = build_vectorstore(
            chunks,
            progress_callback=progress_callback,
        )

        if cache_dir is not None:
            save_vectorstore_cache(vectorstore, cache_dir, content_hash)

    return RAGChain(
        vectorstore=vectorstore,
        filename=", ".join(filenames),
        chunks=chunks,
    )


def process_pdf_and_get_chain(pdf_path: str, cache_dir=None, progress_callback=None):
    """기존 단일 PDF 호출 방식도 계속 지원합니다."""
    return process_pdfs_and_get_chain(
        [pdf_path],
        cache_dir=cache_dir,
        progress_callback=progress_callback,
    )
