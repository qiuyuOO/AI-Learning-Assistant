import requests

article = {
    "title": "Give Your Coding Agents a Memory You Own",
    "description": """
    We introduce a memory system for coding agents.
    The system allows coding agents to persist and retrieve
    information across sessions.
    """,
    "source": "Hugging Face"
}

payload = {
    "model": "jev-latest",
    "state": article,
    "questions": {
        "valuable": {
            "type": "noul",
            "instructions": """
            这篇文章对用户是否具有较高的学习和实践价值？
            用户主要关注 Agent、LLM、RAG、MCP 和 AI 工程。
            如果文章能够帮助用户学习这些领域的技术、
            架构、工程实践或实际应用，则认为有价值。
            """
        }
    }
}

response = requests.post(
    "http://127.0.0.1:8017/v1/systemone",
    json=payload
)

print(response.json())