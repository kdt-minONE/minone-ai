# from fastapi import FastAPI

# from app.core.logging_config import setup_logging

# setup_logging()
# app = FastAPI()


# @app.get("/")
# async def root():
#     return {"message": "Hello World"}


# @app.get("/hello/{name}")
# async def say_hello(name: str):
#     return {"message": f"Hello {name}"}
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from app.ai.agent import build_agent_workflow
from app.ai.state import AgentState
from app.rag.retriever import get_retriever

# --- AI 에이전트 실행 함수 ---
def run_minone_agent(question: str) -> str:
    """
    사용자의 질문을 받아 AI 에이전트 워크플로우를 실행하고,
    최종 답변 또는 다음 행동(재질문)을 문자열로 반환합니다.
    """
    agent = build_agent_workflow()
    initial_state = AgentState(
        question=question, documents=[], answer="",
        assessment_result="", retries=0, messages=[]
    )
    
    print(f"\n{'='*20} 민 ONE 에이전트 실행 시작 {'='*20}")
    print(f"입력된 질문: {question}")
    print(f"{'='*55}\n")

    final_state = None
    for step_output in agent.stream(initial_state, stream_mode="values"):
        final_state = step_output

    print(f"\n{'='*20} 민 ONE 에이전트 실행 종료 {'='*20}")
    
    if final_state:
        final_answer = final_state.get("answer", "")
        # 정형화된 답변이 있다면 그것을 우선 반환
        if "■ 민원 내용:" in final_answer:
            return final_answer
        # 재질문 등 다른 메시지가 있다면 마지막 메시지를 반환
        elif final_state.get("messages"):
            return final_state["messages"][-1].content
        else:
            return "오류: 최종 답변을 생성하지 못했습니다."
    else:
        return "에이전트 실행 중 오류가 발생했습니다."
    
if __name__ == "__main__":
    print("AI 에이전트 로컬 테스트를 시작합니다.")
    
    # 1. Vector Store 준비 (최초 1회 또는 업데이트 시 필요)
    print("Vector Store를 준비합니다...")
    get_retriever()
    print("Vector Store 준비 완료.\n")

    # 2. 테스트 질문 정의
    test_question = "밤 10시에 우리 아파트 앞에서 누가 쓰레기를 무단투기했는데, 어떻게 처리되나요?"
    
    # 3. AI 에이전트 실행
    final_result = run_minone_agent(test_question)
    
    # 4. 최종 결과 출력
    print("\n--- 최종 결과 ---")
    print(final_result)