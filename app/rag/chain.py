# app/rag/chain.py

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from app.rag import prompts
from app.rag.retriever import get_retriever
from app.ai.state import AgentState

# 모델 초기화
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.1)

# 모든 함수의 시그니처가 (state: AgentState) -> dict 형태로 변경

def assess_question_node(state: AgentState) -> dict:
    """1. 질문 분석 노드: 사용자의 질문이 민원 처리에 충분한 정보를 담고 있는지 평가합니다."""
    print("--- 노드 1: 질문 분석 및 정보 충분성 평가 ---")
    question = state['question']
    
    assess_chain = prompts.assess_question_prompt | llm | StrOutputParser()
    assessment_result = assess_chain.invoke({"question": question})
    
    print(f"평가 결과: {assessment_result}")
    
    # 반환값은 업데이트할 상태 필드만 담은 '딕셔너리'
    return {
        "assessment_result": assessment_result,
        "messages": [HumanMessage(content=question)]
    }

def request_clarification_node(state: AgentState) -> dict:
    """2. 추가 정보 요청 노드: 정보가 불충분할 경우, 사용자에게 명확한 질문을 생성합니다."""
    print("--- 노드 2: 추가 정보 요청 (재질문 생성) ---")
    question = state['question']
    
    clarification_chain = prompts.request_clarification_prompt | llm | StrOutputParser()
    clarification_message = clarification_chain.invoke({"question": question})
    
    print(f"생성된 재질문: {clarification_message}")
    
    return {
        "retries": state['retries'] + 1,
        "messages": [AIMessage(content=clarification_message)]
    }


def retrieve_documents_node(state: AgentState) -> dict:
    """3. 문서 검색 노드: Vector Store에서 관련 법령 문서를 검색합니다."""
    print("--- 노드 3: 관련 법령 문서 검색 (RAG) ---")
    question = state['question']
    
    retriever = get_retriever()
    documents = retriever.invoke(question)
    doc_contents = [doc.page_content for doc in documents]
    
    print(f"{len(doc_contents)}개의 관련 문서를 검색했습니다.")
    return {"documents": doc_contents}

def assess_answer_quality_node(state: AgentState) -> dict:
    """4. 답변 품질 평가 노드: 검색된 문서가 답변 생성에 유효한지 평가합니다."""
    print("--- 노드 4: 검색된 문서의 유효성 평가 ---")
    
    quality_assessment_chain = prompts.assess_answer_quality_prompt | llm | StrOutputParser()
    assessment_result = quality_assessment_chain.invoke(state)
    
    print(f"문서 품질 평가 결과: {assessment_result}")
    return {"assessment_result": assessment_result}


def generate_answer_node(state: AgentState) -> dict:
    """5. 답변 초안 생성 노드: 검색된 문서를 바탕으로 답변의 초안을 작성합니다."""
    print("--- 노드 5: 답변 초안 생성 ---")
    
    rag_chain = prompts.generate_answer_prompt | llm | StrOutputParser()
    context = "\n\n---\n\n".join(state['documents'])
    answer = rag_chain.invoke({"context": context, "question": state['question']})
    
    print(f"생성된 답변 초안:\n{answer}")
    return {"answer": answer}

def format_answer_node(state: AgentState) -> dict:
    """6. 최종 답변 정형화 노드: 생성된 초안을 공식 템플릿에 맞춰 최종 답변으로 만듭니다."""
    print("--- 노드 6: 최종 답변 정형화 ---")
    
    format_chain = prompts.format_answer_prompt | llm | StrOutputParser()
    final_answer = format_chain.invoke(state)
    
    print(f"최종 정형화된 답변:\n{final_answer}")
    return {"answer": final_answer, "messages": [AIMessage(content=final_answer)]}