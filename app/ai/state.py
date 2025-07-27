from typing import List, TypedDict, Annotated, Dict, Any
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    """
    LangGraph 에이전트의 상태를 정의하는 TypedDict 입니다.
    각 노드는 이 상태의 일부를 읽고, 일부를 업데이트하여 다음 노드로 전달합니다.

    Attributes:
        question: 사용자의 원본 질문
        cleaned_question: 악성 민원 필터링 및 정제 과정을 거친 후의 핵심 민원 내용.
        documents: RAG를 통해 검색된 관련 법령 문서 목록
        answer: 민원인에게 제공될 답변 또는 최종 보고서 전체
        assistant_answer: LLM이 생성한 답변 초안
        assessment_result: 노드 분기를 위한 판단 결과 (e.g., 'sufficient' or 'insufficient')
        final_report: 민원인용 답변과 담당자용 정보가 모두 포함된 최종 결과물(딕셔셔리)
        retries: 재시도 횟수 (재질문 또는 재검색)
        messages: 전체 대화 기록. `operator.add`를 사용하여 메시지가 덮어쓰이지 않고 계속 추가되도록 합니다.
    """
    question: str
    cleaned_question: str
    documents: List[str]
    answer: str
    assistant_answer: str
    assessment_result: str
    final_report: Dict[str, Any]
    retries: int
    messages: Annotated[List[BaseMessage], operator.add]