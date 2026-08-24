# AI 학습지원 플랫폼

> 교수자가 등록한 PDF 강의자료를 기반으로 학생의 학습과 교수자의 수업 준비를 지원하는 RAG 기반 교육 서비스입니다.

## 배포 서비스

**[AI 학습지원 플랫폼 바로가기](https://comento-ai-education-team-6u3ysphln9jxml5yzen8wl.streamlit.app/)**

## 1. 서비스 개요

교수자는 여러 PDF 강의자료를 등록하고 학생 공개 여부를 설정할 수 있습니다. 학생은 공개된 강의자료를 선택하여 질문하거나 쉬운 설명과 학습 퀴즈를 제공받을 수 있습니다. AI 답변은 선택한 자료에서 검색한 내용을 바탕으로 생성되며, 관련 파일명과 페이지 등 출처를 함께 표시합니다.

### 주요 기능

- 교수자·학생 역할별 로그인 및 화면 분리
- 여러 PDF 강의자료 업로드·선택·삭제
- 강의자료별 학생 공개·비공개 설정
- 선택한 자료 기반 RAG 질의응답
- 강의자료 핵심 내용 요약 및 학생용 쉬운 설명
- OX·단답형 학습 퀴즈와 자동 채점
- 답변 근거 파일명·페이지·본문 일부 표시
- PDF 근거 페이지 미리보기 및 다운로드
- 복수 채팅 생성·전환과 대화 기록 관리
- 한국어·영어 답변 언어 선택
- 근거가 부족한 경우 답변 생성을 제한하는 안전장치

## 2. 팀원별 역할

| 팀원 | 역할 | 담당 업무 |
| --- | --- | --- |
| 김영상 | 팀장 / AI Solutions Architect | 기준 프로토타입 선정, RAG 구조 및 인터페이스 설계, 프롬프트·근거 부족 처리 기준 통합 |
| 최락현 | AI Engineer | RAG 기능 검증, 검색·답변 생성 로직 개선, 예외 처리 및 기능 테스트 |
| 지유성 | Cloud/Infra Engineer | Streamlit Community Cloud 배포, 환경변수·의존성·배포 환경 점검 |
| 오세윤 | Product Owner / QA | 사용자 요구사항과 MVP 범위 관리, 사용자 시나리오 및 기능 검수 |

## 3. 기술 구성

- UI 및 애플리케이션: Streamlit
- LLM 및 임베딩: OpenAI API, LangChain
- 벡터 검색: FAISS
- PDF 처리: PyMuPDF
- 배포: Streamlit Community Cloud
- 협업: GitHub Branch → Pull Request → Review → Merge

자세한 시스템 구조와 예외 처리 기준은 [`docs/architecture.md`](docs/architecture.md)에서 확인할 수 있습니다.

## 4. 로컬 실행 방법

### 4.1 저장소 내려받기

```powershell
git clone https://github.com/comento-AI-education-team-2/comento-AI-education-team.git
cd comento-AI-education-team
```

### 4.2 가상환경 생성 및 실행

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 4.3 패키지 설치

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4.4 API 키 등록

프로젝트 최상위 폴더에 `.env` 파일을 만들고 다음 내용을 입력합니다. 값에 쌍따옴표는 필요하지 않습니다.

```env
OPENAI_API_KEY=발급받은_OpenAI_API_키
```

`.env`와 Streamlit Secrets 파일은 `.gitignore`로 제외되며 GitHub에 올리지 않습니다.

### 4.5 Streamlit 실행

```powershell
python -m streamlit run app.py
```

실행 후 브라우저에서 `http://localhost:8501`로 접속합니다.

## 5. 프로젝트 구조

```text
comento-AI-education-team/
├── app.py                 # Streamlit 화면과 사용자 기능
├── rag_module.py          # PDF 처리, 검색, 요약 및 답변 생성
├── requirements.txt       # 실행에 필요한 Python 패키지
├── .gitignore             # API 키, 가상환경, PDF 등 제외
├── README.md
└── docs/
    └── architecture.md    # RAG 시스템 아키텍처와 운영 기준
```

## 6. 보안 및 운영 유의사항

- OpenAI API 키는 코드나 GitHub 저장소에 직접 기록하지 않습니다.
- 현재 로그인 계정은 MVP 시연용이며 실제 서비스에서는 별도의 인증 시스템으로 교체해야 합니다.
- Streamlit Community Cloud의 로컬 저장소는 재시작 시 초기화될 수 있으므로, 운영 환경에서는 외부 데이터베이스 또는 오브젝트 스토리지 연동이 필요합니다.
- 업로드한 PDF는 저작권과 개인정보 처리 기준을 확인한 자료만 사용해야 합니다.
