"""
Streamlit Frontend for JobYaari RAG Chatbot
===========================================

This module provides a user-friendly web interface for the JobYaari RAG chatbot.
It communicates with the FastAPI backend to process user queries and display responses.

Features:
- Clean, intuitive chat interface
- Real-time query processing
- Source document display
- Error handling and feedback
- Configuration display
"""

import os
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime
import uuid

# Streamlit imports
import streamlit as st
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Environment
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('frontend.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
API_BASE_URL = os.getenv("AGENT_API_BASE_URL", "http://localhost:8080")
REQUEST_TIMEOUT = int(os.getenv("AGENT_REQUEST_TIMEOUT", "30"))

class ChatbotClient:
    """Client for communicating with the FastAPI backend"""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url.rstrip('/')
        self.session = self._create_session()
        logger.info(f"Initialized ChatbotClient with base URL: {self.base_url}")
    
    def _create_session(self):
        """Create a requests session with retry strategy"""
        session = requests.Session()
        
        # Retry strategy
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            #method_whitelist=["HEAD", "GET", "OPTIONS", "POST"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def health_check(self) -> Dict:
        """Check if the API is healthy"""
        try:
            response = self.session.get(
                f"{self.base_url}/health",
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise
    
    def get_info(self) -> Dict:
        """Get service information"""
        try:
            response = self.session.get(
                f"{self.base_url}/info",
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get info: {e}")
            raise
    
    def chat(self, query: str, session_id: Optional[str] = None) -> Dict:
        """Send chat request to the API"""
        try:
            payload = {
                "query": query,
                "session_id": session_id
            }
            
            response = self.session.post(
                f"{self.base_url}/chat",
                json=payload,
                timeout=REQUEST_TIMEOUT,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Chat request failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in chat: {e}")
            raise

def init_streamlit_config():
    """Initialize Streamlit configuration"""
    st.set_page_config(
        page_title="JobYaari RAG Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def init_session_state():
    """Initialize Streamlit session state variables"""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "client" not in st.session_state:
        st.session_state.client = ChatbotClient()
    
    if "api_status" not in st.session_state:
        st.session_state.api_status = "unknown"

def check_api_status():
    """Check and update API status"""
    try:
        st.session_state.client.health_check()
        st.session_state.api_status = "healthy"
        return True
    except Exception as e:
        st.session_state.api_status = f"unhealthy: {str(e)}"
        return False

def display_sidebar():
    """Display sidebar with system information and controls"""
    with st.sidebar:
        st.header("🎯 JobYaari Chatbot")
        st.markdown("---")
        
        # API Status
        st.subheader("🔌 API Status")
        if check_api_status():
            st.success("Connected")
        else:
            st.error(f"Disconnected: {st.session_state.api_status}")
        
        st.markdown("---")
        
        # System Information
        st.subheader("ℹ️ System Info")
        try:
            info = st.session_state.client.get_info()
            st.write(f"**Version:** {info.get('version', 'Unknown')}")
            st.write(f"**Service:** {info.get('service_name', 'Unknown')}")
            
            # Configuration
            config = info.get('config', {})
            st.write(f"**LLM Provider:** {config.get('llm_provider', 'Unknown')}")
            st.write(f"**Embedding:** {config.get('embedding_provider', 'Unknown')}")
            st.write(f"**Collection:** {config.get('qdrant_collection', 'Unknown')}")
            
        except Exception as e:
            st.error(f"Failed to load system info: {str(e)}")
        
        st.markdown("---")
        
        # Usage Instructions
        st.subheader("💡 How to Use")
        st.markdown("""
        **Example Queries:**
        - "Latest Engineering jobs"
        - "Science jobs with 1 year experience"
        - "Tell me about Photo Type Setter position"
        - "Show me Commerce category jobs"
        """)
        
        st.markdown("---")
        
        # Session Info
        st.subheader("🔍 Session Info")
        st.write(f"**Session ID:** {st.session_state.session_id[:8]}...")
        st.write(f"**Messages:** {len(st.session_state.messages)}")
        
        # Clear Chat Button
        if st.button("🗑️ Clear Chat", help="Clear all chat messages"):
            st.session_state.messages = []
            st.rerun()

def display_message(role: str, content: str, sources: Optional[List[Dict]] = None, metadata: Optional[Dict] = None):
    """Display a chat message with proper formatting"""
    with st.chat_message(role):
        st.markdown(content)
        
        # Display sources if available
        if sources and len(sources) > 0:
            with st.expander(f"📄 Sources ({len(sources)} found)", expanded=False):
                for i, source in enumerate(sources, 1):
                    st.markdown(f"""
                    **{i}. {source.get('title', 'Unknown Position')}**  
                    Company: {source.get('company', 'Unknown')}  
                    Score: {source.get('score', 0):.3f}  
                    Link: {source.get('link', 'N/A')}
                    """)
        
        # Display metadata if available and in debug mode
        if metadata and st.session_state.get('debug_mode', False):
            with st.expander("🔧 Debug Info", expanded=False):
                st.json(metadata)

def process_query(query: str):
    """Process user query and get response"""
    try:
        # Show spinner while processing
        with st.spinner("🤔 Thinking..."):
            response = st.session_state.client.chat(
                query=query,
                session_id=st.session_state.session_id
            )
        
        # Add messages to session state
        st.session_state.messages.append({
            "role": "user",
            "content": query,
            "timestamp": datetime.now().isoformat()
        })
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response.get("answer", "No response generated"),
            "sources": response.get("sources", []),
            "metadata": response.get("metadata", {}),
            "timestamp": datetime.now().isoformat()
        })
        
        return True
        
    except requests.exceptions.ConnectionError:
        st.error("❌ Could not connect to the chatbot service. Please check if the server is running.")
        return False
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out. The server might be busy. Please try again.")
        return False
    except requests.exceptions.RequestException as e:
        st.error(f"🔥 Request failed: {str(e)}")
        return False
    except Exception as e:
        st.error(f"💥 An unexpected error occurred: {str(e)}")
        logger.error(f"Unexpected error in process_query: {e}")
        return False

def main():
    """Main Streamlit application"""
    
    # Initialize configuration and session state
    init_streamlit_config()
    init_session_state()
    
    # Display sidebar
    display_sidebar()
    
    # Main content area
    st.title("🤖 JobYaari RAG Chatbot")
    st.markdown("Ask me about job postings, requirements, and opportunities!")
    
    # Check API status in main area if unhealthy
    if st.session_state.api_status != "healthy":
        st.error(f"⚠️ API Status: {st.session_state.api_status}")
        st.info("Please check the sidebar for more information and ensure the FastAPI server is running.")
    
    # Display chat messages
    for message in st.session_state.messages:
        display_message(
            role=message["role"],
            content=message["content"],
            sources=message.get("sources"),
            metadata=message.get("metadata")
        )
    
    # Chat input
    if query := st.chat_input("Ask about jobs, qualifications, companies, or anything else..."):
        # Display user message immediately
        display_message("user", query)
        
        # Process query and get response
        if process_query(query):
            # Display assistant response
            last_message = st.session_state.messages[-1]
            if last_message["role"] == "assistant":
                display_message(
                    role="assistant",
                    content=last_message["content"],
                    sources=last_message.get("sources"),
                    metadata=last_message.get("metadata")
                )
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🔧 Debug Mode**")
        debug_mode = st.checkbox("Enable debug info", value=False)
        st.session_state.debug_mode = debug_mode
    
    with col2:
        st.markdown("**📊 Statistics**")
        total_messages = len(st.session_state.messages)
        user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
        st.write(f"Total: {total_messages} | Queries: {user_messages}")
    
    with col3:
        st.markdown("**⚡ Performance**")
        if st.session_state.messages:
            st.write("Session active")
        else:
            st.write("New session")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application error: {e}")
        
        # Show error details in debug mode
        if st.session_state.get('debug_mode', False):
            st.exception(e)