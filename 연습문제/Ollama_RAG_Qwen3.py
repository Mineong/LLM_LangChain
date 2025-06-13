from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.chains import RetrievalQA

print("==> 1. 문서 로딩 → PDF 읽기...")
loader = PyPDFLoader('./data/tutorial-korean.pdf')
documents = loader.load()
print(f"  총 {len(documents)}페이지 로드 완료")

print("==> 2. 문서 분할 → 작은 청크로 나누기")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ".", " ", ""]
)
chunks = text_splitter.split_documents(documents)
print(f"  {len(chunks)}개 청크 생성 완료")

print("==> 3. 벡터화 → 임베딩으로 변환")
embeddings = OllamaEmbeddings(
    model="qwen3:1.7b",
    base_url="http://localhost:11434"
)

print("==> 4. 저장 → FAISS 벡터스토어에 저장")
vectorstore = FAISS.from_documents(chunks, embeddings)

print("===> 5. 검색 → 질문과 유사한 문서 찾기")
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

print("===> 6. 생성 → LLM으로 답변 생성")
llm = ChatOllama(
    model="qwen3:1.7b",
    base_url="http://localhost:11434",
    temperature=0.1,
    num_predict=1500
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

query = "이 문서의 핵심 내용은 무엇인가요?"
result = qa_chain.invoke(query)

print("\n[🔍 질의 결과]")
print(result["result"])
print("\n[📄 참고 문서]")
for doc in result["source_documents"]:
    print(f"- {doc.metadata['source']}")
