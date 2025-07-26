import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

VECTOR_STORE_PATH = "data/vector_store/faiss_index"
LAW_DATA_PATH = "data/laws"

def batch_faiss_build(splits, embeddings, batch_size=100):
    """splits를 여러 FAISS 인덱스로 나눠 생성 후 병합"""
    vectorstores = []
    for i in range(0, len(splits), batch_size):
        batch = splits[i:i+batch_size]
        vs = FAISS.from_documents(batch, embeddings)
        vectorstores.append(vs)
    main_vs = vectorstores[0]
    for vs in vectorstores[1:]:
        main_vs.merge_from(vs)
    return main_vs

def get_retriever():
    """
    저장된 Vector Store를 로드하여 리트리버를 반환합니다.
    만약 Vector Store가 없다면 새로 생성합니다.
    """
    if not os.path.exists(VECTOR_STORE_PATH):
        print("저장된 Vector Store가 없어 새로 생성합니다.")
        # 1. 문서 로드 (data/laws 폴더의 모든 PDF)
        loader = PyPDFDirectoryLoader(LAW_DATA_PATH)
        documents = loader.load()
        print(f"총 {len(documents)}개의 PDF 문서를 로드했습니다.")

        # 2. 텍스트 분할
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        splits = text_splitter.split_documents(documents)
        print(f"문서를 총 {len(splits)}개로 분할했습니다.")

        # 3. 임베딩 및 Vector Store 생성 (배치 처리) 추후 large model로 변경 가능
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_store = batch_faiss_build(splits, embeddings, batch_size=100)

        # 4. 로컬에 저장
        vector_store.save_local(VECTOR_STORE_PATH)
        print(f"Vector Store를 '{VECTOR_STORE_PATH}' 경로에 저장했습니다.")
    else:
        print(f"'{VECTOR_STORE_PATH}' 경로에서 기존 Vector Store를 로드합니다.")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)

    # 검색기(Retriever) 반환 (가장 유사한 문서 5개 검색)
    return vector_store.as_retriever(search_kwargs={'k': 5})

#python app/rag/retriever.py
if __name__ == '__main__':
    retriever = get_retriever()
    print("\n리트리버가 성공적으로 로드되었습니다.")
    test_query = "민원 처리 기간에 대해 알려줘"
    results = retriever.invoke(test_query)
    print(f"\n테스트 쿼리 '{test_query}'에 대한 검색 결과:")
    for doc in results:
        print(f"- (Source: {doc.metadata.get('source', 'N/A')}) {doc.page_content[:100]}...")
