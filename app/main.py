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
    

# 로컬 테스트
if __name__ == "__main__":
    print("Vector Store를 준비합니다...")
    get_retriever()
    print("Vector Store 준비 완료.\n")

    # --- 대화형 테스트 루프 ---
    
    # 1. 첫 질문을 받습니다.
    initial_question = input("🙋 사용자: ")
    
    # 2. 대화 기록을 저장할 리스트를 만듭니다.
    conversation_history = [initial_question]

    while True:
        # 3. 현재까지의 대화 내용을 하나의 질문으로 합칩니다.
        #    이렇게 해야 AI가 이전 대화의 맥락을 이해할 수 있습니다.
        full_question = "\n".join(conversation_history)
        
        # 4. AI 에이전트를 실행합니다.
        agent_response = run_minone_agent(full_question)
        
        print(f"\n🤖 AI: {agent_response}")
        
        # 5. AI의 답변을 확인하고 루프를 계속할지 결정합니다.
        #    만약 최종 답변 템플릿이 포함되어 있다면, 대화가 끝난 것입니다.
        if "■ 민원 내용:" in agent_response:
            print("\n--- 최종 답변이 생성되어 대화를 종료합니다. ---")
            break
        
        # 6. AI가 추가 질문을 했으므로, 사용자의 다음 답변을 입력받습니다.
        user_response = input("\n🙋 사용자: ")
        
        # 7. 사용자의 답변을 대화 기록에 추가하고 다시 루프를 시작합니다.
        conversation_history.append(user_response)