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
    retries = state.get("retries", 0)
    
    print(f"문서 품질 평가 결과: {assessment_result}")
    
    return {
        "assessment_result": assessment_result,
        "retries": retries
    }


def generate_answer_node(state: AgentState) -> dict:
    """5. 답변 초안 생성 노드: 검색된 문서를 바탕으로 답변의 초안을 작성합니다."""
    print("--- 노드 5: 답변 초안 생성 ---")
    
    rag_chain = prompts.generate_answer_prompt | llm | StrOutputParser()
    context = "\n\n---\n\n".join(state['documents'])
    answer = rag_chain.invoke({"context": context, "question": state['question']})
    
    print(f"생성된 답변 초안:\n{answer}")
    # 'assistant_answer' 필드에 초안을 저장
    return {"assistant_answer": answer}

def filter_and_sanitize_node(state: AgentState) -> dict:
    """6. 민원 필터링 및 정제 노드: 담당자가 볼 수 있도록 원본 질문을 정제합니다."""
    print("---  노드 6: 민원 내용 필터링 및 정제 ---")
    question = state['question']
    
    sanitize_chain = prompts.filter_and_sanitize_prompt | llm | StrOutputParser()
    cleaned_question = sanitize_chain.invoke({"question": question})
    
    print(f"정제된 민원 내용: {cleaned_question}")
    return {"cleaned_question": cleaned_question}

def create_final_report_node(state: AgentState) -> dict:
    """7. 최종 보고서 생성 노드: 모든 정보를 취합하여 최종 결과물을 생성합니다."""
    print("--- 노드 7: 최종 보고서 생성 ---")
    
    # 이전 단계의 `generate_answer_node`에서 생성한 답변 초안을 사용합니다.
    report_chain = prompts.create_final_report_prompt | llm | StrOutputParser()
    final_report_str = report_chain.invoke({
        "question": state['question'],
        "answer": state['assistant_answer'],
        "cleaned_question": state['cleaned_question']
    })
    
    print(f"최종 생성된 보고서:\n{final_report_str}")
    # 최종 결과물을 'answer' 필드에 저장하여 출력을 통일합니다.
    return {"answer": final_report_str}