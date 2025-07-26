from langgraph.graph import StateGraph, END
from app.ai.state import AgentState
from app.rag.chain import (
    assess_question_node,
    request_clarification_node,
    retrieve_documents_node,
    assess_answer_quality_node,
    generate_answer_node,
    format_answer_node,
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
            return "request_clarification"

def route_after_quality_assessment(state: AgentState) -> str:
    """
    '답변 품질 평가' 노드 실행 후, 검색된 문서가 유효한지에 따라 다음 경로를 결정합니다.
    - sufficient: 품질 충분 -> 답변 생성 단계로 진행
    - insufficient: 품질 불충분 -> 재검색 또는 (횟수 초과 시) 강제 진행
    """
    assessment_result = state.get("assessment_result", "").lower()

    if "sufficient" in assessment_result:
        print("✅경로 결정: 문서 품질 충분. '답변 생성'으로 이동합니다.")
        return "generate_answer"
    else:
        # 재검색 횟수를 확인합니다.
        if state.get('retries', 0) >= MAX_RETRIES:
            print(f"⚠️ 경로 결정: 최대 재검색 횟수({MAX_RETRIES}회) 도달. '답변 생성'을 강제 실행합니다.")
            return "generate_answer"  # 실패하더라도 일단 답변 생성을 시도합니다.
        else:
            print("▶️ 경로 결정: 문서 품질 불충분. '문서 검색'을 다시 시도합니다.")
            # 재시도 횟수를 여기서 직접 증가시켜도 되지만, 노드에서 처리하는 것이 더 명확할 수 있습니다.
            # 여기서는 재검색 경로를 타도록 지시만 합니다.
            state['retries'] += 1
            return "retrieve_documents"


def build_agent_workflow():
    """
    LangGraph 워크플로우를 정의하고 모든 노드와 엣지를 연결한 후,
    컴파일된 에이전트(그래프)를 반환합니다.
    """
    workflow = StateGraph(AgentState)

    # 1. 노드 정의: 그래프에 참여할 모든 작업(함수)을 이름과 함께 등록합니다.
    workflow.add_node("assess_question", assess_question_node)
    workflow.add_node("request_clarification", request_clarification_node)
    workflow.add_node("retrieve_documents", retrieve_documents_node)
    workflow.add_node("assess_answer_quality", assess_answer_quality_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("format_answer", format_answer_node)

    # 2. 엣지 연결: 노드 간의 작업 흐름(방향)을 정의합니다.

    # 2-1. 시작점 설정: 에이전트는 항상 'assess_question' 노드에서 시작합니다.
    workflow.set_entry_point("assess_question")

    # 2-2. '질문 분석' 후의 조건부 분기
    workflow.add_conditional_edges(
        "assess_question",
        route_after_question_assessment,
        {
            # 라우팅 함수의 반환값에 따라 실행될 노드를 매핑합니다.
            "request_clarification": "request_clarification",
            "retrieve_documents": "retrieve_documents",
        },
    )
    
    # 2-3. '추가 정보 요청' 후에는 사용자의 다음 입력을 기다려야 하므로, 현재 그래프는 여기서 종료됩니다.
    # 실제 챗봇 애플리케이션에서는 이 부분에서 사용자 입력을 받는 루프를 구현하게 됩니다.
    workflow.add_edge("request_clarification", END)
    
    # 2-4. '문서 검색' 후에는 항상 '답변 품질 평가'를 수행합니다.
    workflow.add_edge("retrieve_documents", "assess_answer_quality")

    # 2-5. '답변 품질 평가' 후의 조건부 분기
    workflow.add_conditional_edges(
        "assess_answer_quality",
        route_after_quality_assessment,
        {
            # 품질이 낮으면 다시 '문서 검색' 노드로 돌아가 재검색을 수행합니다.
            "retrieve_documents": "retrieve_documents",
            # 품질이 충분하면 '답변 생성' 노드로 이동합니다.
            "generate_answer": "generate_answer",
        },
    )
    
    # 2-6. '답변 생성' 후에는 '최종 답변 정형화'를 수행합니다.
    workflow.add_edge("generate_answer", "format_answer")
    
    # 2-7. '최종 답변 정형화' 후에는 모든 작업이 완료되었으므로 그래프를 종료합니다.
    workflow.add_edge("format_answer", END)

    # 3. 그래프 컴파일: 정의된 모든 노드와 엣지를 바탕으로 실행 가능한 에이전트를 생성합니다.
    print("LangGraph 워크플로우를 컴파일합니다...")
    agent = workflow.compile()
    print("✅ 에이전트 컴파일 완료!")
    return agent