from langchain_community.utilities import GoogleSerperAPIWrapper
load_dotenv()

def web_search_tool(query: str) -> str:
    Tool(
        name="Intermediate Answer",
        func=search.run,
        description="useful for when you need to ask with search"
    )
    return f"Search results for: {query}"
