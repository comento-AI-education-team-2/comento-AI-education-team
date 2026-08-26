import os
import re
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
import pymupdf as fitz
from pydantic import BaseModel, Field

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


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


# =========================================================
# 1. PDF를 페이지별 Document로 읽기
# =========================================================
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


# =========================================================
# 2. 페이지를 청크로 분할
# =========================================================
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
    )

    # split_documents를 사용하면 원본 metadata가 청크에도 유지됩니다.
    return splitter.split_documents(documents)


# =========================================================
# 3. 임베딩 + FAISS
# =========================================================
def build_vectorstore(chunks):
    if not chunks:
        raise ValueError(
            "PDF에서 읽을 수 있는 텍스트를 찾지 못했습니다."
        )

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    return FAISS.from_documents(
        chunks,
        embedding=embeddings,
    )


# =========================================================
# 4. 프롬프트
# =========================================================
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
3. 공식은 가능한 경우 Markdown 수식 표기법을 사용하세요.
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


# =========================================================
# 5. 짧은 질문용 핵심어 추출
# =========================================================
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

    # 요약 표현을 제외한 실제 주제어가 남으면 일반 질의로 처리합니다.
    # 예: "단면계수를 요약해줘"는 단면계수 검색을 수행합니다.
    return not extract_search_keywords(normalized_question)


# =========================================================
# 6. 실제 RAG 객체
# =========================================================
class RAGChain:
    def __init__(self, vectorstore, filename, chunks):
        self.vectorstore = vectorstore
        self.filename = filename
        self.chunks = chunks

        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
        )

        self.prompt = get_prompt()
        self.quiz_prompt = get_quiz_prompt()
        self.summary_prompt = get_summary_prompt()
        self.quiz_llm = self.llm.with_structured_output(QuizSet)

    def compare_retrieval_thresholds(
        self,
        question: str,
        thresholds=(0.15, 0.25, 0.35),
        k: int = 8,
    ):
        """같은 검색 결과를 여러 유사도 임계값으로 비교합니다.

        답변 생성 모델은 호출하지 않고 검색 결과만 비교하므로,
        임계값 조정 전 평가용으로 사용할 수 있습니다.
        """
        normalized_question = question.strip()

        if not normalized_question:
            raise ValueError("비교할 질문을 입력해주세요.")

        if is_broad_summary_request(normalized_question):
            return {
                "broad_summary": True,
                "question": normalized_question,
                "comparisons": [],
                "keyword_hits": 0,
            }

        normalized_thresholds = sorted({
            float(threshold)
            for threshold in thresholds
            if 0.0 <= float(threshold) <= 1.0
        })

        if not normalized_thresholds:
            raise ValueError("0과 1 사이의 임계값이 필요합니다.")

        results = (
            self.vectorstore
            .similarity_search_with_relevance_scores(
                normalized_question,
                k=k,
            )
        )
        results = sorted(
            results,
            key=lambda item: float(item[1]),
            reverse=True,
        )

        keywords = extract_search_keywords(normalized_question)
        keyword_hits = 0

        if keywords:
            keyword_hits = sum(
                1
                for doc in self.chunks
                if any(
                    keyword in doc.page_content.lower()
                    for keyword in keywords
                )
            )

        comparisons = []

        for threshold in normalized_thresholds:
            matched_results = [
                (doc, score)
                for doc, score in results
                if float(score) >= threshold
            ]
            evidence = []

            for doc, score in matched_results[:3]:
                excerpt = " ".join(doc.page_content.split())
                evidence.append({
                    "file": doc.metadata.get("source", self.filename),
                    "page": doc.metadata.get("page", "?"),
                    "score": round(float(score), 3),
                    "excerpt": excerpt[:180],
                })

            comparisons.append({
                "threshold": threshold,
                "matched_count": len(matched_results),
                "top_score": (
                    round(float(results[0][1]), 3)
                    if results
                    else None
                ),
                "evidence": evidence,
            })

        return {
            "broad_summary": False,
            "question": normalized_question,
            "comparisons": comparisons,
            "keyword_hits": keyword_hits,
        }

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

            # 자료의 앞부분에만 치우치지 않도록 전체 구간에서
            # 대표 청크도 함께 선택합니다.
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
            "answer": response.content,
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

        # 일반 질문의 점수 기준을 적용하지 않고, 선택한 각 파일에서
        # 퀴즈에 쓸 대표 근거를 따로 가져옵니다.
        for source_name in source_names:
            documents = self.vectorstore.similarity_search(
                search_query,
                k=3,
                filter={"source": source_name},
            )

            # 벡터 저장소 버전에 따라 필터 결과가 없을 때를 대비한
            # 안전한 대체 선택입니다.
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

        # "전체를 요약해줘"처럼 특정 검색어가 없는 요청은
        # 유사도 검색 기준을 적용하지 않고 파일별 대표 내용을
        # 고르게 선택하는 전용 요약 흐름으로 보냅니다.
        if is_broad_summary_request(question):
            return self.generate_summary()

        # ---------------------------------------------
        # 검색 + 유사도 점수
        # ---------------------------------------------
        results = (
            self.vectorstore
            .similarity_search_with_relevance_scores(
                question,
                k=8,
            )
        )

        # ---------------------------------------------
        # 근거 부족 검사
        # ---------------------------------------------
        #
        # 주의:
        # 의미검색 점수가 낮더라도 PDF에 질문 핵심어가 직접
        # 등장하면 아래 핵심어 검색에서 근거로 다시 포함합니다.
        # 실제 PDF/질문 세트로 테스트 후 조정해야 합니다.
        #
        threshold = 0.15

        relevant_results = [
            (doc, score)
            for doc, score in results
            if score >= threshold
        ]

        # ---------------------------------------------
        # 짧은 질문용 핵심어 직접 검색
        # ---------------------------------------------
        # 예: "단면계수가 뭐야" → "단면계수"
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

                if not any(
                    keyword in content
                    for keyword in keywords
                ):
                    continue

                chunk_key = (
                    doc.metadata.get("source"),
                    doc.metadata.get("page"),
                    doc.page_content,
                )

                if chunk_key not in existing_chunks:
                    # 직접 일치한 근거임을 나타내는 내부 점수
                    relevant_results.append((doc, 1.0))
                    existing_chunks.add(chunk_key)

                if len(relevant_results) >= 8:
                    break

        if not relevant_results:
            return {
                "answer": (
                    "선택한 강의자료에서 충분한 "
                    "근거를 찾지 못했습니다."
                ),
                "sources": [],
                "has_evidence": False,
            }

        # ---------------------------------------------
        # 검색된 문서를 Context로 구성
        # ---------------------------------------------
        context_parts = []

        for doc, score in relevant_results:
            source = doc.metadata.get(
                "source",
                self.filename,
            )
            page = doc.metadata.get("page", "?")

            context_parts.append(
                f"[출처: {source}, 페이지: {page}]\n"
                f"{doc.page_content}"
            )

        context = "\n\n".join(context_parts)

        # ---------------------------------------------
        # LLM 호출
        # ---------------------------------------------
        prompt_question = question

        if answer_instruction.strip():
            prompt_question = (
                f"{question}\n\n"
                "[답변 작성 언어 지침]\n"
                f"{answer_instruction.strip()}"
            )

        messages = self.prompt.format_messages(
            context=context,
            question=prompt_question,
        )

        response = self.llm.invoke(messages)

        answer = response.content

        # ---------------------------------------------
        # 출처 정리
        # ---------------------------------------------
        sources = []
        seen = set()

        for doc, score in relevant_results:
            source = doc.metadata.get(
                "source",
                self.filename,
            )
            page = doc.metadata.get("page", "?")

            key = (source, page)

            if key not in seen:
                seen.add(key)

                sources.append(
                    {
                        "file": source,
                        "page": page,
                        "score": round(float(score), 3),
                        "excerpt": doc.page_content.strip(),
                    }
                )

        return {
            "answer": answer,
            "sources": sources,
            "has_evidence": True,
        }


# =========================================================
# 7. 여러 PDF → 하나의 RAG 준비
# =========================================================
def process_pdfs_and_get_chain(pdf_paths):
    documents = []
    filenames = []

    for pdf_path in pdf_paths:
        documents.extend(
            load_pdf_documents(pdf_path)
        )
        filenames.append(Path(pdf_path).name)

    chunks = split_documents(documents)
    vectorstore = build_vectorstore(chunks)

    return RAGChain(
        vectorstore=vectorstore,
        filename=", ".join(filenames),
        chunks=chunks,
    )


def process_pdf_and_get_chain(pdf_path: str):
    """기존 단일 PDF 호출 방식도 계속 지원합니다."""
    return process_pdfs_and_get_chain([pdf_path])
