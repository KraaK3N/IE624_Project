"""Department Chatbot Core Logic."""

from typing import List, Dict
from src.config import MODEL_NAME


class DepartmentChatbot:
    """Chatbot for department queries, FAQs, and course guidance."""

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name

    def respond(self, query: str, history: List[Dict[str, str]] = None) -> str:
        """Generate response for user query.
        
        TODO: Plug in your RAG pipeline or LLM client here.
        """
        # Minimal placeholder response for initialization
        if not query.strip():
            return "How can I help you with department inquiries today?"
            
        return (
            f"[Echo / Demo Mode]: Received your question: \"{query}\". "
            "Please configure your LLM API key and connect your department knowledge base in `src/bot.py`."
        )
