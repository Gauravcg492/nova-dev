import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from agent.tool_rag import build_vectorstore, create_rag_tool
from langchain_community.utilities import GoogleSerperAPIWrapper
from typing import Optional

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Initialize LLM (same provider wrapper used elsewhere in this repo)
agent_llm = ChatOpenAI(
    model="google/gemini-2.5-flash-preview-09-2025",
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

# Optional RAG: if you set RAG_PDF_PATH in the environment, we will try to build/load
# a vectorstore and create a retrieval function. If not set or load fails, RAG is skipped.
RAG_PDF_PATH = os.getenv("RAG_PDF_PATH")
_rag_fn = None
if RAG_PDF_PATH:
    try:
        _vectorstore = build_vectorstore(RAG_PDF_PATH)
        _rag_fn = create_rag_tool(_vectorstore)
    except Exception as e:
        print(f"Warning: failed to build/load RAG vectorstore: {e}")
        _rag_fn = None


def web_search_text(query: str) -> str:
    """Run a simple web search and return a textual summary.

    Uses GoogleSerperAPIWrapper directly instead of creating a LangChain Tool class.
    """
    try:
        os.environ["SERPER_API_KEY"] = os.getenv("SERPER_DEV_API") or os.environ.get("SERPER_API_KEY", "")
        wrapper = GoogleSerperAPIWrapper()
        return wrapper.run(query)
    except Exception as e:
        return f"[web search failed: {e}]"


def _build_prompt(video_analysis_report: str, user_question: str, rag_context: Optional[str], web_context: Optional[str]) -> str:
    parts = [
        "You are a baby safety expert assistant. Use the provided materials to answer the user's question.",
        "\n---\n",
        "VIDEO_ANALYSIS_REPORT:\n",
        video_analysis_report or "(no report provided)",
        "\n---\n",
        "USER_QUESTION:\n",
        user_question,
    ]

    if rag_context:
        parts.extend(["\n---\n", "INTERNAL_DOC_CONTEXT:\n", rag_context])

    if web_context:
        parts.extend(["\n---\n", "WEB_SEARCH_RESULTS:\n", web_context])

    parts.extend([
        "\n---\n",
        "Return the answer in the following format:\n",
        "- OVERALL_RISK: one of LOW, MEDIUM, HIGH, CRITICAL\n",
        "- SUMMARY: short description of the main safety issues\n",
        "- ACTION_STEPS: a numbered list of clear corrections the caregiver should make\n",
    ])

    return "\n".join(parts)


def get_baby_safety_answer(video_analysis_report: str, user_question: str, use_rag: bool = True, use_web: bool = False) -> str:
    """Produce a baby safety answer by optionally adding RAG and/or web search context,
    then calling the LLM directly. This avoids LangChain Agent tooling.

    Args:
        video_analysis_report: the text report produced by the video analyzer
        user_question: the human's question
        use_rag: whether to include internal PDF RAG context (if available)
        use_web: whether to include the web search pass

    Returns:
        The raw LLM text output.
    """

    rag_context = None
    web_context = None

    if use_rag and _rag_fn:
        try:
            rag_context = _rag_fn(user_question)
        except Exception as e:
            rag_context = f"[RAG retrieval failed: {e}]"

    if use_web:
        web_context = web_search_text(user_question)

    prompt = _build_prompt(video_analysis_report, user_question, rag_context, web_context)

    system_msg = SystemMessage(content="You are an expert baby safety assistant. Be concise, factual and provide clear action steps.")
    human_msg = HumanMessage(content=prompt)

    try:
        response = agent_llm.invoke([system_msg, human_msg])
        return response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        return f"[LLM call failed: {e}]"


if __name__ == "__main__":
    sample_report = "DETECTED_ISSUE: Baby head tilt forward. SEVERITY: MEDIUM."
    q = "What should the caregiver change to improve head/neck support?"
    print(get_baby_safety_answer(sample_report, q, use_rag=False, use_web=False))