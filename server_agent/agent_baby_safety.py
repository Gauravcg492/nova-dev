import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import Tool, initialize_agent, AgentType
from tool_rag import build_vectorstore, create_rag_tool
from tool_web_search import web_search

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# llm
agent_llm = ChatOpenAI(model="google/gemini-2.5-flash-preview-09-2025",  
                api_key=OPENROUTER_API_KEY,
                base_url="https://openrouter.ai/api/v1"
            )

# tools
vectorstore = build_vectorstore(r"C:\Users\manig\Downloads\nova_rag.pdf")
rag_fn = create_rag_tool(vectorstore)
rag_tool = Tool(
    name="BabySafetyRAG",
    func=rag_fn,
    description=(
        "Use this tool to retrieve baby safety information from the internal PDF knowledge base. "
        "Input should be a natural language question about baby safety, sleep, swaddling, etc."
    ),
)
web_tool = web_search()  
tools = [rag_tool, web_tool]

agent = initialize_agent(
    tools=tools,
    llm=agent_llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

def get_baby_safety_answer(video_analysis_report: str, user_question: str) -> str:
    prompt = f"""
    You are a baby safety expert assistant.

    You receive:
    1) A video safety analysis report from a specialist model
    2) A human question

    VIDEO_ANALYSIS_REPORT:
    {video_analysis_report}

    USER_QUESTION:
    {user_question}

    Use the report as your primary observation.
    Use tools only if needed:
    - BabySafetyRAG for internal PDF guidelines
    - Intermediate Answer by Web Search for external sources

    Return:
    - OVERALL_RISK: one of LOW, MEDIUM, HIGH, CRITICAL
    - SUMMARY: short description of the main safety issues
    - ACTION_STEPS: a numbered list of clear corrections the caregiver should make
        """.strip()

    return agent.run(prompt)