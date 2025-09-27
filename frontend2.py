"""
Simplified Streamlit frontend for JobYaari RAG Chatbot
Keeps the same backend routes: /health, /info, /chat

How to run:
1. Install dependencies: pip install streamlit requests
2. Run: streamlit run streamlit_frontend_simplified.py

This is a minimal, easy-to-read single-file Streamlit app.
"""

import os
import requests
from typing import Optional, Dict
import streamlit as st

# Default backend URL (can be changed in sidebar)
DEFAULT_API_BASE = os.getenv("API_BASE_URL", "http://localhost:8080")
REQUEST_TIMEOUT = 10


class ChatbotClient:
    """Minimal client for the FastAPI backend using the same routes."""

    def __init__(self, base_url: str = DEFAULT_API_BASE):
        self.base_url = base_url.rstrip('/')

    def health(self) -> Dict:
        resp = requests.get(f"{self.base_url}/health", timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()

    def info(self) -> Dict:
        resp = requests.get(f"{self.base_url}/info", timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()

    def chat(self, query: str, session_id: Optional[str] = None) -> Dict:
        payload = {"query": query, "session_id": session_id}
        resp = requests.post(f"{self.base_url}/chat", json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()


def init_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []  # list of (role, text)
    if 'session_id' not in st.session_state:
        st.session_state.session_id = None


def main():
    st.set_page_config(page_title="JobYaari - Simple Chat", layout="wide")
    st.title("JobYaari — Simplified RAG Chatfront")

    # Sidebar: backend URL and simple controls
    st.sidebar.header("Backend settings")
    api_base = st.sidebar.text_input("API base URL", value=DEFAULT_API_BASE)
    client = ChatbotClient(api_base)

    # Health check
    try:
        health = client.health()
        st.sidebar.success(f"Backend healthy: {health}")
    except Exception as e:
        st.sidebar.error(f"Backend unreachable: {e}")
        st.warning("Backend is unreachable. Make sure the FastAPI server is running on the configured URL and port.")

    # Info (optional)
    if st.sidebar.button("Fetch /info"):
        try:
            info = client.info()
            st.sidebar.json(info)
        except Exception as e:
            st.sidebar.error(f"Failed to fetch /info: {e}")

    init_session_state()

    # Main chat UI
    with st.form(key='chat_form', clear_on_submit=False):
        query = st.text_area("Your question", height=120)
        submitted = st.form_submit_button("Send")

    if submitted and query.strip():
        st.session_state.messages.append(("user", query.strip()))
        try:
            result = client.chat(query.strip(), session_id=st.session_state.session_id)
            # Expecting JSON with keys like 'answer', 'session_id', maybe 'sources'
            answer = result.get('answer') if isinstance(result, dict) else result
            st.session_state.session_id = result.get('session_id') if isinstance(result, dict) and result.get('session_id') else st.session_state.session_id

            if isinstance(answer, (dict, list)):
                # display structured response
                st.session_state.messages.append(("assistant", answer))
            else:
                st.session_state.messages.append(("assistant", str(answer)))

        except Exception as e:
            st.error(f"Chat request failed: {e}")

    # Render chat history
    for role, text in st.session_state.messages:
        if role == 'user':
            st.markdown(f"**You:** {text}")
        else:
            # If it's structured, show JSON
            if isinstance(text, (dict, list)):
                st.markdown("**Assistant:**")
                st.json(text)
            else:
                st.markdown(f"**Assistant:** {text}")

    # Clear chat
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.session_id = None
        st.experimental_rerun()


if __name__ == '__main__':
    main()
