import gradio as gr
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import os

# 기본 모델 경로 설정
DEFAULT_MODEL_PATH = "/home/edentns/tasha/axolotl/axolotl/outputs/out"
# FAISS 인덱스 저장 경로
INDEX_DIR = "faiss_index"

def get_device_info():
    """GPU 사용 가능 여부와 정보를 확인합니다."""
    if torch.cuda.is_available():
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**2  # MB
        memory_reserved = torch.cuda.memory_reserved(0) / 1024**2    # MB
        return f"GPU 사용 가능: {device_name}\n할당된 메모리: {memory_allocated:.2f}MB\n예약된 메모리: {memory_reserved:.2f}MB"
    else:
        return "GPU를 사용할 수 없습니다. CPU를 사용합니다."

class QwenEmbedding:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).eval()
    
    def embed_documents(self, texts):
        """문서들을 임베딩합니다."""
        embeddings = []
        for text in texts:
            embedding = self.embed_query(text)
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text):
        """단일 쿼리를 임베딩합니다."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # 마지막 hidden state의 평균을 임베딩으로 사용
            last_hidden = outputs.hidden_states[-1]
            embedding = last_hidden.mean(dim=1).squeeze().cpu().numpy()
        return embedding

def extract_text_from_docx(file):
    """Word 문서에서 텍스트를 추출합니다."""
    doc = Document(file.name)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

def split_by_articles(text):
    """조항 단위로 텍스트를 분할합니다."""
    # 조항 패턴: "제X조" 또는 "제 X 조" 형식
    article_pattern = r'제\s*\d+\s*조'
    
    # 조항을 찾아서 분할
    articles = re.split(f'({article_pattern})', text)
    
    # 조항과 내용을 결합
    result = []
    for i in range(0, len(articles)-1, 2):
        if i+1 < len(articles):
            article = articles[i] + articles[i+1]
            if article.strip():
                result.append(article.strip())
    
    return result

def save_index(index, chunks, embeddings, filename):
    """FAISS 인덱스와 메타데이터를 저장합니다."""
    os.makedirs(INDEX_DIR, exist_ok=True)
    
    # FAISS 인덱스 저장
    faiss.write_index(index, os.path.join(INDEX_DIR, f"{filename}.index"))
    
    # 메타데이터 저장
    metadata = {
        "chunks": chunks,
        "embeddings_shape": embeddings.shape
    }
    with open(os.path.join(INDEX_DIR, f"{filename}.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

def load_index(filename):
    """FAISS 인덱스와 메타데이터를 로드합니다."""
    index_path = os.path.join(INDEX_DIR, f"{filename}.index")
    metadata_path = os.path.join(INDEX_DIR, f"{filename}.json")
    
    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        return None, None, None
    
    # FAISS 인덱스 로드
    index = faiss.read_index(index_path)
    
    # 메타데이터 로드
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    return index, metadata["chunks"], metadata["embeddings_shape"]

def list_saved_indices():
    """저장된 인덱스 목록을 반환합니다."""
    if not os.path.exists(INDEX_DIR):
        return []
    
    indices = []
    for file in os.listdir(INDEX_DIR):
        if file.endswith(".index"):
            indices.append(file[:-6])  # .index 확장자 제거
    return indices

def process_document(file, model_path=DEFAULT_MODEL_PATH):
    """문서를 처리하고 임베딩을 생성합니다."""
    # GPU 정보 확인
    device_info = get_device_info()
    
    # 텍스트 추출
    text = extract_text_from_docx(file)
    
    # 조항 단위로 분할
    articles = split_by_articles(text)
    
    # 각 조항 내에서 세부 분할
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False
    )
    
    chunks = []
    for article in articles:
        # 조항이 너무 길 경우 세부 분할
        if len(article) > 1000:
            sub_chunks = text_splitter.split_text(article)
            chunks.extend(sub_chunks)
        else:
            chunks.append(article)
    
    # 임베딩 생성
    embeddings = QwenEmbedding(model_path)
    chunk_embeddings = embeddings.embed_documents(chunks)
    chunk_embeddings = np.array(chunk_embeddings)
    
    # FAISS 인덱스 생성
    dimension = chunk_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(chunk_embeddings.astype('float32'))
    
    # 인덱스 저장
    filename = os.path.splitext(os.path.basename(file.name))[0]
    save_index(index, chunks, chunk_embeddings, filename)
    
    return f"{device_info}\n\n문서가 성공적으로 처리되었습니다.", chunks, index, embeddings

def answer_question(question, index, embeddings, chunks):
    """질문에 대한 답변을 생성합니다."""
    if not question:
        return "질문을 입력해주세요."
    
    # 질문 임베딩
    question_embedding = embeddings.embed_query(question)
    
    # 유사한 문서 검색
    D, I = index.search(np.array([question_embedding]).astype('float32'), k=3)
    
    # 관련 컨텍스트 추출
    relevant_chunks = [chunks[i] for i in I[0]]
    context = "\n".join(relevant_chunks)
    
    return f"관련 컨텍스트:\n\n{context}"

def view_index_contents(index_name):
    """저장된 인덱스의 내용을 확인합니다."""
    index, chunks, embeddings_shape = load_index(index_name)
    if index is None:
        return "인덱스를 찾을 수 없습니다."
    
    # 인덱스 정보
    info = f"인덱스 크기: {index.ntotal} 벡터\n"
    info += f"임베딩 차원: {embeddings_shape[1]}\n\n"
    
    # 청크 내용
    info += "저장된 청크:\n"
    for i, chunk in enumerate(chunks):
        info += f"\n[{i}] {chunk[:200]}..." if len(chunk) > 200 else f"\n[{i}] {chunk}"
    
    return info

# Gradio 인터페이스 생성
with gr.Blocks() as demo:
    gr.Markdown("# 문서 Q&A 시스템")
    
    with gr.Row():
        with gr.Column():
            model_path = gr.Textbox(label="모델 경로", value=DEFAULT_MODEL_PATH)
            file_input = gr.File(label="Word 문서 업로드")
            process_btn = gr.Button("문서 처리")
            status_output = gr.Textbox(label="처리 상태")
            chunks_output = gr.Textbox(label="문서 청크", lines=10)
    
    with gr.Row():
        with gr.Column():
            question_input = gr.Textbox(label="질문")
            answer_btn = gr.Button("답변 생성")
            answer_output = gr.Textbox(label="답변", lines=10)
    
    with gr.Row():
        with gr.Column():
            index_name = gr.Dropdown(label="저장된 인덱스", choices=list_saved_indices())
            view_btn = gr.Button("인덱스 내용 보기")
            index_contents = gr.Textbox(label="인덱스 내용", lines=10)
    
    # 상태 저장을 위한 변수들
    index = gr.State(None)
    embeddings = gr.State(None)
    chunks = gr.State(None)
    
    # 이벤트 핸들러
    process_btn.click(
        process_document,
        inputs=[file_input, model_path],
        outputs=[status_output, chunks_output, index, embeddings]
    )
    
    answer_btn.click(
        answer_question,
        inputs=[question_input, index, embeddings, chunks_output],
        outputs=[answer_output]
    )
    
    view_btn.click(
        view_index_contents,
        inputs=[index_name],
        outputs=[index_contents]
    )

if __name__ == "__main__":
    demo.launch() 