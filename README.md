 구현 흐름
PDF → 텍스트 추출 (PyMuPDF)

텍스트 → Chunk 분할 (LangChain TextSplitter)

Chunk → 임베딩 → FAISS 저장

사용자가 질문하면 → embedding → FAISS → 유사한 문서 context 추출

LLM으로 QA 처리 → 사용자에게 응답