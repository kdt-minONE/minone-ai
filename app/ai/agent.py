from langgraph.graph import StateGraph, END
from app.ai.state import AgentState
from app.rag.chain import (
    assess_question_node,
    request_clarification_node,
    retrieve_documents_node,
    assess_answer_quality_node,
    generate_answer_node,
    filter_and_sanitize_node,
    create_final_report_node,
)

# 최대 재시도 횟수 설정 (재질문, 재검색에 각각 적용)
MAX_RETRIES = 1


def route_after_question_assessment(state: AgentState) -> str:
    """
    '질문 분석' 노드 실행 후, 정보가 충분한지에 따라 다음 경로를 결정합니다.
    - sufficient: 정보 충분 -> 문서 검색 단계로 진행
    - insufficient: 정보 불충분 -> 재질문 또는 (횟수 초과 시) 강제 진행
    """
    assessment_result = state.get("assessment_result", "").lower().strip()
    
    if assessment_result == "sufficient":
        print("✅ 경로 결정: 정보 충분. '문서 검색'으로 이동합니다.")
        # 다음 단계를 위해 재시도 횟수를 초기화합니다.
        state['retries'] = 0
        return "retrieve_documents"
    else:
        # 재질문 횟수를 확인합니다.
        if state.get('retries', 0) >= MAX_RETRIES:
            print(f"⚠️ 경로 결정: 최대 재질문 횟수({MAX_RETRIES}회) 도달. '문서 검색'을 강제 실행합니다.")
            state['retries'] = 0 # 재검색을 위해 카운터 초기화
            return "retrieve_documents"
        else:
            print("▶️ 경로 결정: 정보 불충분. '추가 정보 요청'으로 이동합니다.")
            # 재질문 시도 횟수를 여기서 직접 업데이트합니다.
            state['retries'] = state.get('retries', 0) + 1
            return "request_clarification"

def route_after_quality_assessment(state: AgentState) -> str:
    """
    '답변 품질 평가' 노드 실행 후, 검색된 문서가 유효한지에 따라 다음 경로를 결정합니다.
    - sufficient: 품질 충분 -> 답변 생성 단계로 진행
    - insufficient: 품질 불충분 -> 재검색 또는 (횟수 초과 시) 강제 진행
    """
    assessment_result = state.get("assessment_result", "").lower()

    if assessment_result == "sufficient":
        print("✅ 경로 결정: 문서 품질 충분. '답변 생성'으로 이동합니다.")
        return "generate_answer"
    else:
        # 재검색 횟수를 확인합니다.
        if state.get('retries', 0) >= MAX_RETRIES:
            print(f"⚠️ 경로 결정: 최대 재검색 횟수({MAX_RETRIES}회) 도달. '답변 생성'을 강제 실행합니다.")
            return "generate_answer"
        else:
            print(f"▶️ 경로 결정: 문서 품질 불충분. '문서 검색'을 다시 시도합니다. (시도 {state.get('retries', 0) + 1}/{MAX_RETRIES})")
            state['retries'] = state.get('retries', 0) + 1
            return "retrieve_documents"


def build_agent_workflow():
    """
    LangGraph 워크플로우를 정의하고 모든 노드와 엣지를 연결한 후,
    컴파일된 에이전트(그래프)를 반환합니다.
    """
    workflow = StateGraph(AgentState)

    # 1. 노드 정의
    workflow.add_node("assess_question", assess_question_node)
    workflow.add_node("request_clarification", request_clarification_node)
    workflow.add_node("retrieve_documents", retrieve_documents_node)
    workflow.add_node("assess_answer_quality", assess_answer_quality_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("filter_and_sanitize", filter_and_sanitize_node)
    workflow.add_node("create_final_report", create_final_report_node)

    # 2. 엣지 연결
    workflow.set_entry_point("assess_question")

    # '질문 분석' 후의 조건부 분기
    workflow.add_conditional_edges(
        "assess_question",
        route_after_question_assessment,
        {"request_clarification": "request_clarification", "retrieve_documents": "retrieve_documents"},
    )
    workflow.add_edge("request_clarification", END)
    
    # '문서 검색' 후 -> '품질 평가'
    workflow.add_edge("retrieve_documents", "assess_answer_quality")

    # '품질 평가' 후의 조건부 분기
    workflow.add_conditional_edges(
        "assess_answer_quality",
        route_after_quality_assessment,
        {"retrieve_documents": "retrieve_documents", "generate_answer": "generate_answer"},
    )
    
    # '답변 생성' 후 -> '민원 내용 정제' (두 작업은 병렬로도 가능하지만, 순차적으로 진행하여 흐름을 명확히 함)
    workflow.add_edge("generate_answer", "filter_and_sanitize")
    
    # '민원 내용 정제' 후 -> '최종 보고서 생성'
    workflow.add_edge("filter_and_sanitize", "create_final_report")
    
    # '최종 보고서 생성' 후 그래프 종료
    workflow.add_edge("create_final_report", END)

    # 3. 그래프 컴파일
    print("🤖 LangGraph 워크플로우를 컴파일합니다...")
    agent = workflow.compile()
    print("✅ 에이전트 컴파일 완료!")
    return agent