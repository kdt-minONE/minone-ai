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
from dotenv import load_dotenv
from app.ai.agent import build_agent_workflow
from app.ai.state import AgentState
from app.rag.retriever import get_retriever

# .env 파일 로드 (OpenAI API 키 등)
load_dotenv()

# --- AI 에이전트 실행 함수 ---
def run_minone_agent(question: str) -> str:
    """
    사용자의 질문을 받아 AI 에이전트 워크플로우를 실행하고,
    최종 답변 또는 다음 행동(재질문)을 문자열로 반환합니다.
    """
    agent = build_agent_workflow()
    
    # ▼▼▼ initial_state 정의 수정 ▼▼▼
    # AgentState에 정의된 모든 필드를 명시적으로 초기화해주는 것이
    # 나중에 상태 관련 버그를 방지하는 데 도움이 됩니다.
    initial_state = AgentState(
        question=question,
        cleaned_question="",
        documents=[],
        answer="",
        assistant_answer="",
        final_report={},
        assessment_result="",
        retries=0,
        messages=[]
    )
    
    print(f"\n{'='*20} 민 ONE 에이전트 실행 시작 {'='*20}")
    print(f"입력된 질문: {question}")
    print(f"{'='*55}\n")

    final_state = None
    for step_output in agent.stream(initial_state, stream_mode="values"):
        current_node = list(step_output.keys())[-1]
        print(f"--- 🏃 현재 실행 노드: {current_node} ---")
        final_state = step_output

    print(f"\n{'='*20} 민 ONE 에이전트 실행 종료 {'='*20}")
    
    if final_state:
        # 최종 결과물은 항상 'answer' 필드에 저장되도록 통일했습니다.
        final_result = final_state.get("answer", "")
        if final_result and "■" in final_result: # 최종 보고서 형식인지 확인
            return final_result
        # 재질문 등 다른 메시지가 있다면 마지막 메시지를 반환
        elif final_state.get("messages"):
            return final_state["messages"][-1].content
        else:
            return "오류: 최종 답변을 생성하지 못했습니다."
    else:
        return "에이전트 실행 중 오류가 발생했습니다."


# --- 대화형 테스트용 코드 ---
# 이 부분은 실제 API 서버가 아닌, 로컬에서 에이전트를 테스트
if __name__ == "__main__":
    print("AI 에이전트 로컬 테스트를 시작합니다.")
    
    print("Vector Store를 준비합니다...")
    get_retriever()
    print("Vector Store 준비 완료.\n")

    # --- 대화형 테스트 루프 ---
    
    print("대화를 시작합니다. 종료하려면 'exit' 또는 'quit'을 입력하세요.")
    # 1. 첫 질문을 받습니다.
    initial_question = input("🙋 사용자: ")
    
    if initial_question.lower() in ["exit", "quit"]:
        print("대화를 종료합니다.")
    else:
        # 2. 대화 기록을 저장할 리스트를 만듭니다.
        conversation_history = [f"사용자: {initial_question}"]

        while True:
            # 3. 현재까지의 대화 내용을 하나의 질문으로 합칩니다.
            full_question = "\n".join(conversation_history)
            
            # 4. AI 에이전트를 실행합니다.
            agent_response = run_minone_agent(full_question)
            
            print(f"\n🤖 AI: {agent_response}")
            
            # 5. AI의 답변을 확인하고 루프를 계속할지 결정합니다.
            if "■" in agent_response:
                print("\n--- 최종 답변이 생성되어 대화를 종료합니다. ---")
                break
            
            # 6. 사용자의 다음 답변을 입력받습니다.
            user_response = input("\n🙋 사용자: ")
            
            if user_response.lower() in ["exit", "quit"]:
                print("대화를 종료합니다.")
                break
            
            # 7. 사용자의 답변을 대화 기록에 추가하고 다시 루프를 시작합니다.
            conversation_history.append(f"사용자: {user_response}")
            conversation_history.append(f"AI: {agent_response}") # AI의 이전 질문도 맥락에 포함