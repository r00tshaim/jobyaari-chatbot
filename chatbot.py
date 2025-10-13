"""
JobYaari RAG Chatbot using LangGraph
====================================

This module implements a Retrieval-Augmented Generation (RAG) chatbot
for JobYaari job postings using LangGraph's StateGraph workflow.

Key Components:
- Query processing and rewriting
- Vector search with Qdrant
- Metadata filtering
- Response generation with configurable LLMs
- Structured output formatting
"""

import os
import logging
from typing import Dict, List, Optional, TypedDict, Annotated
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo  # Add this import at the top if not present
import re

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

# Qdrant imports
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Environment and utilities
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GraphState(TypedDict):
    """State for the RAG workflow graph"""
    query: str
    rewritten_query: Optional[str]
    category_filter: Optional[str]
    experience_filter: Optional[str]
    retrieved_docs: List[Dict]
    final_answer: str
    error_message: Optional[str]

@dataclass 
class ChatbotConfig:
    """Configuration class for the RAG chatbot"""
    # LLM Configuration
    llm_provider: str = os.getenv("AGENT_LLM_PROVIDER", "gemini")
    gemini_model: str = os.getenv("AGENT_GEMINI_MODEL", "gemini-2.0-flash-exp")
    openai_model: str = os.getenv("AGENT_OPENAI_MODEL", "gpt-3.5-turbo")
    gemini_api_key: str = os.getenv("AGENT_GEMINI_API_KEY", "")
    openai_api_key: str = os.getenv("AGENT_OPENAI_API_KEY", "")
    
    # Embedding Configuration
    embedding_provider: str = os.getenv("AGENT_EMBEDDING_PROVIDER", "ollama")
    ollama_host: str = os.getenv("AGENT_OLLAMA_HOST", "http://localhost:11434")
    embedding_model: str = os.getenv("AGENT_EMBEDDING_MODEL", "nomic-embed-text:latest")
    
    # Qdrant Configuration
    qdrant_url: str = os.getenv("AGENT_QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str = os.getenv("AGENT_QDRANT_API_KEY", "")
    qdrant_collection: str = os.getenv("AGENT_QDRANT_COLLECTION", "jobyaari_jobs")
    
    # Retrieval Configuration
    max_results: int = int(os.getenv("AGENT_MAX_RESULTS", "5"))
    similarity_threshold: float = float(os.getenv("AGENT_SIMILARITY_THRESHOLD", "0.7"))

class JobYaariRAGChatbot:
    """RAG Chatbot implementation using LangGraph StateGraph"""
    
    def __init__(self, config: Optional[ChatbotConfig] = None):
        """Initialize the chatbot with configuration"""
        self.config = config or ChatbotConfig()
        logger.info("Initializing JobYaari RAG Chatbot with LangGraph")
        
        # Initialize components
        self.llm = self._init_llm()
        self.embeddings = self._init_embeddings()
        self.qdrant_client = self._init_qdrant_client()
        self.vector_store = self._init_vector_store()
        
        # Build the LangGraph workflow
        self.workflow = self._build_workflow()
        
        logger.info("RAG Chatbot initialized successfully")
    
    def _init_llm(self):
        """Initialize the LLM based on configuration"""
        try:
            if self.config.llm_provider.lower() == "gemini":
                if not self.config.gemini_api_key:
                    raise ValueError("Gemini API key not provided")
                logger.info(f"Initializing Gemini LLM: {self.config.gemini_model}")
                return ChatGoogleGenerativeAI(
                    model=self.config.gemini_model,
                    google_api_key=self.config.gemini_api_key,
                    temperature=0.1
                )
            elif self.config.llm_provider.lower() == "openai":
                if not self.config.openai_api_key:
                    raise ValueError("OpenAI API key not provided")
                logger.info(f"Initializing OpenAI LLM: {self.config.openai_model}")
                return ChatOpenAI(
                    model=self.config.openai_model,
                    api_key=self.config.openai_api_key,
                    temperature=0.1
                )
            else:
                raise ValueError(f"Unsupported LLM provider: {self.config.llm_provider}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def _init_embeddings(self):
        """Initialize embeddings model"""
        try:
            if self.config.embedding_provider.lower() == "ollama":
                logger.info(f"Initializing Ollama embeddings: {self.config.embedding_model}")
                return OllamaEmbeddings(
                    model=self.config.embedding_model,
                    base_url=self.config.ollama_host
                )
            elif self.config.embedding_provider.lower() == "gemini" or self.config.embedding_provider.lower() == "google":
                if not self.config.gemini_api_key:
                    raise ValueError("Gemini API key not provided for embeddings")
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                logger.info(f"Initializing Google Generative AI (Gemini) embeddings: {self.config.embedding_model}")
                return GoogleGenerativeAIEmbeddings(
                    model=self.config.embedding_model,
                    google_api_key=self.config.gemini_api_key
                )
            else:
                raise ValueError(f"Unsupported embedding provider: {self.config.embedding_provider}")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise
    
    def _init_qdrant_client(self):
        """Initialize Qdrant client"""
        try:
            logger.info(f"Connecting to Qdrant at: {self.config.qdrant_url}")
            if self.config.qdrant_api_key:
                client = QdrantClient(
                    url=self.config.qdrant_url,
                    api_key=self.config.qdrant_api_key
                )
            else:
                client = QdrantClient(url=self.config.qdrant_url)
            
            # Test connection
            collections = client.get_collections()
            logger.info(f"Successfully connected to Qdrant. Collections: {len(collections.collections)}")
            return client
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise
    
    def _init_vector_store(self):
        """Initialize Qdrant vector store"""
        try:
            vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=self.config.qdrant_collection,
                embedding=self.embeddings,
            )
            logger.info(f"Initialized vector store for collection: {self.config.qdrant_collection}")
            return vector_store
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for RAG"""
        logger.info("Building LangGraph workflow")
        
        # Create StateGraph
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("query_analysis", self._analyze_query)
        workflow.add_node("retrieve", self._retrieve_documents)
        workflow.add_node("generate", self._generate_response)
        
        # Add edges
        workflow.add_edge(START, "query_analysis")
        workflow.add_edge("query_analysis", "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        
        # Compile the workflow
        compiled_workflow = workflow.compile()
        logger.info("LangGraph workflow compiled successfully")
        
        return compiled_workflow
    
    def _analyze_query(self, state: GraphState) -> GraphState:
        """Analyze and rewrite query for better retrieval"""
        logger.info(f"Analyzing query: {state['query']}")
        
        try:
            #Query analysis prompt (extended)
            analysis_prompt = ChatPromptTemplate.from_template("""
            You are a job search assistant. Analyze the user query to extract:
            1. Rewritten query for better similarity search in vector DB
            2. Category filter (if mentioned: Engineering, Science, Commerce, etc.)
            3. Experience filter (if mentioned: like "1 year", "2 years", etc.)
            4. State filter (explicit state name like "Arunachal Pradesh", "Maharastra", "Gujarat", etc or NONE)
            5. Education filter (like "Any Degree", "B.Sc", "BTech", etc., or NONE)
            6. Minimum salary (only the number, e.g. 20000 or NONE)
            7. Region (like "North East India" or NONE)

            User Query: {query}

            Provide your analysis in this exact format (one per line):
            REWRITTEN_QUERY: [improved search query for simiarity search in vector DB]
            CATEGORY_FILTER: [category name or NONE]
            EXPERIENCE_FILTER: [experience requirement or NONE]
            STATE_FILTER: [state name or NONE]
            EDUCATION_FILTER: [education requirement or NONE]
            SALARY_MIN: [numeric minimum salary or NONE]
            REGION_FILTER: [region name or NONE]
            """)
            
            # Get analysis from LLM
            chain = analysis_prompt | self.llm | StrOutputParser()
            analysis = chain.invoke({"query": state["query"]})
            
            # Parse analysis
            rewritten_query = None
            category_filter = None
            experience_filter = None
            state_filter = None
            education_filter = None
            salary_min = None
            region_filter = None

            for line in analysis.strip().split('\n'):
                if line.startswith('REWRITTEN_QUERY:'):
                    rewritten_query = line.split(':', 1)[1].strip()
                elif line.startswith('CATEGORY_FILTER:'):
                    cat = line.split(':', 1)[1].strip()
                    category_filter = cat if cat != "NONE" else None
                elif line.startswith('EXPERIENCE_FILTER:'):
                    exp = line.split(':', 1)[1].strip()
                    experience_filter = exp if exp != "NONE" else None
                elif line.startswith('STATE_FILTER:'):
                    st = line.split(':', 1)[1].strip()
                    state_filter = st if st != "NONE" else None
                elif line.startswith('EDUCATION_FILTER:'):
                    ed = line.split(':', 1)[1].strip()
                    education_filter = ed if ed != "NONE" else None
                elif line.startswith('SALARY_MIN:'):
                    sal = line.split(':', 1)[1].strip()
                    try:
                        salary_min = int(sal.replace(',', '')) if sal != "NONE" and sal != "" else None
                    except:
                        salary_min = None
                elif line.startswith('REGION_FILTER:'):
                    rg = line.split(':', 1)[1].strip()
                    region_filter = rg if rg != "NONE" else None
            
            # Update state (added new keys)
            state["rewritten_query"] = rewritten_query or state["query"]
            state["category_filter"] = category_filter
            state["experience_filter"] = experience_filter
            state["state_filter"] = state_filter
            state["education_filter"] = education_filter
            state["salary_min"] = salary_min
            state["region_filter"] = region_filter
            
            logger.info(f"Query analysis complete - Rewritten Query: {state['rewritten_query']},  category: {state['category_filter']}, experience: {state['experience_filter']}, state: {state['state_filter']}, education: {state['education_filter']}, salary_min: {state['salary_min']}, region: {state['region_filter']}")
            
        except Exception as e:
            logger.error(f"Error in query analysis: {e}")
            state["rewritten_query"] = state["query"]
            state["error_message"] = f"Query analysis failed: {str(e)}"
        
        return state
    
    def _retrieve_documents(self, state: GraphState) -> GraphState:
        """Retrieve relevant documents from Qdrant"""
        #query = state.get("rewritten_query", state["query"])
        query = state['query'] # Use original query for retrieval to keep it simple
        logger.info(f"Retrieving documents for: {query}")
        
        try:
            # Build Qdrant filter based on analysis (category, experience, state, education, region)
            qdrant_filter = None
            filter_conditions = []
            should_conditions = []

            if state.get("category_filter"):
                filter_conditions.append(
                    FieldCondition(
                        key="category",
                        match=MatchValue(value=state["category_filter"])
                    )
                )

            if state.get("experience_filter"):
                # filter_conditions.append(
                #     FieldCondition(
                #         key="experience",
                #         match=MatchValue(value=state["experience_filter"])
                #     )
                # )
                pass  # Keep as is for now, need to fix expreince metadata format in Qdrant

            # exact state filter (e.g. "Arunachal Pradesh")
            if state.get("state_filter"):
                filter_conditions.append(
                    FieldCondition(
                        key="state",
                        match=MatchValue(value=state["state_filter"])
                    )
                )

            # education filter (exact match / tag)
            if state.get("education_filter"):
                filter_conditions.append(
                    FieldCondition(
                        key="education",
                        match=MatchValue(value=state["education_filter"])
                    )
                )

            # region -> map to a list of states (simple mapping for "North East India")
            region = state.get("region_filter")
            if region:
                # Normalize
                rg = region.strip().lower()
                if "north" in rg and "east" in rg:
                    ne_states = [
                        "Arunachal Pradesh", "Assam", "Manipur", "Meghalaya",
                        "Mizoram", "Nagaland", "Tripura", "Sikkim"
                    ]
                    for s in ne_states:
                        should_conditions.append(
                            FieldCondition(
                                key="state",
                                match=MatchValue(value=s)
                            )
                        )
                # other region mappings can be added here

            # Compose final filter
            if filter_conditions and should_conditions:
                qdrant_filter = Filter(must=filter_conditions, should=should_conditions)
            elif filter_conditions:
                qdrant_filter = Filter(must=filter_conditions)
            elif should_conditions:
                qdrant_filter = Filter(should=should_conditions)

            # If salary_min present, increase k to fetch more candidates for post-filtering (hybrid approach)
            search_kwargs = {
                "k": self.config.max_results,
                "score_threshold": self.config.similarity_threshold
            }
            if state.get("salary_min"):
                # fetch more so that we can post-filter by numeric salary afterward
                search_kwargs["k"] = max(self.config.max_results * 3, self.config.max_results)

            if qdrant_filter:
                # Convert qdrant model objects to plain dict for LangChain/Qdrant compatibility.
                # Some langchain/Qdrant wrappers expect primitive dicts for `filter`.
                try:
                    search_kwargs["filter"] = qdrant_filter.dict()
                except Exception:
                    try:
                        search_kwargs["filter"] = qdrant_filter.to_dict()
                    except Exception:
                        # Fallback: pass-through (keeps original behavior if conversion not available)
                        search_kwargs["filter"] = qdrant_filter
                logger.info(
                    "qdrant_filter -> type: %s, value (truncated): %s",
                    type(search_kwargs["filter"]),
                    str(search_kwargs["filter"])[:500],
                )

            # Log search parameters before creating retriever (helps debug k/score_threshold/filter)
            logger.info("Search kwargs before as_retriever: %s", search_kwargs)
            # Get retriever and search
            retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs=search_kwargs
            )
            
            retrieved_docs = retriever.invoke(query)
            logger.info(f"****Shaim raw retrieved_docs: {retrieved_docs}")
            
            formatted_docs = []
            for doc in retrieved_docs:
                doc_dict = {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(getattr(doc, 'score', 0.0)) if getattr(doc, 'score', None) is not None else 0.0
                }
                formatted_docs.append(doc_dict)

            # Helper: parse numeric salary from metadata salary_range (very simple)
            def _parse_salary(salary_str):
                if not salary_str:
                    return None
                # find first numeric group
                m = re.search(r'(\d[\d,]*)', str(salary_str))
                if not m:
                    return None
                try:
                    return int(m.group(1).replace(',', ''))
                except:
                    return None

            final_docs = formatted_docs

            # If salary_min provided, post-filter by numeric salary (hybrid approach)
            salary_min = state.get("salary_min")
            if salary_min:
                filtered = []
                for d in formatted_docs:
                    md = d.get("metadata", {})
                    sal_num = _parse_salary(md.get("salary_range") or md.get("salary") or "")
                    if sal_num is not None and sal_num >= salary_min:
                        filtered.append(d)
                final_docs = filtered

            # Boost exact category matches (small additive boost)
            cat_req = state.get("category_filter")
            if cat_req:
                for d in final_docs:
                    md = d.get("metadata", {})
                    if md.get("category") and str(md.get("category")).lower() == str(cat_req).lower():
                        d["score"] = d.get("score", 0.0) + 0.25

            # Sort by score (desc) and trim to max_results
            final_docs = sorted(final_docs, key=lambda x: x.get("score", 0.0), reverse=True)
            final_docs = final_docs[: self.config.max_results]

            state["retrieved_docs"] = final_docs
            logger.info(f"Retrieved {len(final_docs)} documents (after metadata filtering/boosting)")
            logger.info(f"****Shaim Retrieved Docs: {final_docs}")

            
        except Exception as e:
            logger.error(f"Error in document retrieval: {e}")
            state["retrieved_docs"] = []
            state["error_message"] = f"Document retrieval failed: {str(e)}"
        
        return state
    
    def _generate_response(self, state: GraphState) -> GraphState:
        """Generate final response using retrieved documents"""
        logger.info("Generating final response")
        
        try:
            retrieved_docs = state.get("retrieved_docs", [])
            
            if not retrieved_docs:
                state["final_answer"] = "I couldn't find any relevant job postings for your query. Please try rephrasing your question or check if there are jobs available in the specified category."
                return state
            
            # Prepare context from retrieved documents
            context_parts = []
            for i, doc in enumerate(retrieved_docs, 1):
                metadata = doc.get("metadata", {})
                content = doc["content"]
                
                # Extract key information from metadata
                job_title = metadata.get("job_title", "Unknown Position")
                company = metadata.get("department_company", "Unknown Company") 
                education = metadata.get("education", "Not specified")
                experience = metadata.get("experience", "Not specified")
                posted_on = metadata.get("posted_on", "Not specified")

               # Extract all important URLs
                official_notice = metadata.get("official_notice_link", "")
                official_website = metadata.get("official_website", "")
                job_detail_url = metadata.get("job_detail_url", "")
                category_url = metadata.get("category_url", "")

                # Format URLs as clickable Markdown links
                official_notice_link = f"[Official Notice]({official_notice})" if official_notice else "No official notice"
                website_link = f"[Official Website]({official_website})" if official_website else "No website"
                job_detail_link = f"[Job Details]({job_detail_url})" if job_detail_url else "No details"
                category_link = f"[Category Page]({category_url})" if category_url else "No category page"
                
                context_part = f"""
                Job {i}:
                Title: {job_title}
                Company: {company}
                Education: {education}
                Experience: {experience}
                Posted: {posted_on}
                Official Notice: {official_notice_link}
                Website: {website_link}
                Job Details: {job_detail_link}
                Category: {category_link}
                Content: {content[:500]}...
                """
                context_parts.append(context_part)
            
            context = "\n".join(context_parts)

            current_date = str(datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%B %d, %Y"))
            
            # Response generation prompt
            response_prompt = ChatPromptTemplate.from_template("""
            You are a helpful JobYaari assistant. Based on the retrieved job postings, provide a structured and informative answer to the user's query.
            
            User Query: {query}
            
            Retrieved Job Postings:
            {context}
            
            Instructions:
            1. Provide a clear, structured response
            2. Include job titles, companies, and key requirements
            3. Include official links when available
            4. If multiple jobs match, list them in a organized way
            5. Be helpful and informative
            6. Use bullet points or numbered lists for job cards
            7. Check job deadlines (last_date) against current date/time ({current_date}) and note if expired (e.g., "This job deadline has passed")
            8. If no relevant jobs are found, politely inform the user
            9. If current date is needed, use this {current_date} as system date for reference (include time for urgency checks)
            
            Response:
            """)
            
            # Generate response
            chain = response_prompt | self.llm | StrOutputParser()
            response = chain.invoke({
                "query": state["query"],
                "context": context,
                "current_date": current_date
            })
            
            state["final_answer"] = response
            logger.info("Response generated successfully")
            
        except Exception as e:
            logger.error(f"Error in response generation: {e}")
            state["final_answer"] = f"I encountered an error while generating the response: {str(e)}"
            state["error_message"] = f"Response generation failed: {str(e)}"
        
        return state
    
    def chat(self, query: str) -> Dict:
        """Main chat interface"""
        logger.info(f"Processing chat query: {query}")
        
        try:
            # Initialize state
            initial_state = {
                "query": query,
                "rewritten_query": None,
                "category_filter": None,
                "experience_filter": None,
                "retrieved_docs": [],
                "final_answer": "",
                "error_message": None
            }
            
            # Run the workflow
            final_state = self.workflow.invoke(initial_state)
            
            # Prepare response
            response = {
                "answer": final_state["final_answer"],
                "sources": [],
                "metadata": {
                    "rewritten_query": final_state.get("rewritten_query"),
                    "category_filter": final_state.get("category_filter"),
                    "experience_filter": final_state.get("experience_filter"),
                    "num_docs_retrieved": len(final_state.get("retrieved_docs", [])),
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            # Add sources if available
            for doc in final_state.get("retrieved_docs", []):
                metadata = doc["metadata"]
                source = {
                    "title": metadata.get("job_title", "Unknown"),
                    "company": metadata.get("department_company", "Unknown"),
                    "link": metadata.get("official_notice_link", ""),
                    "score": doc.get("score", 0.0)
                }
                response["sources"].append(source)
            
            # Add error if any
            if final_state.get("error_message"):
                response["error"] = final_state["error_message"]
            
            logger.info("Chat processing completed successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error in chat processing: {e}")
            return {
                "answer": f"I encountered an error while processing your query: {str(e)}",
                "sources": [],
                "metadata": {"error": str(e), "timestamp": datetime.now().isoformat()}
            }

# Global chatbot instance
_chatbot_instance = None

def get_chatbot() -> JobYaariRAGChatbot:
    """Get singleton chatbot instance"""
    global _chatbot_instance
    if _chatbot_instance is None:
        _chatbot_instance = JobYaariRAGChatbot()
    return _chatbot_instance

def chat_with_jobs(query: str) -> Dict:
    """Simple interface for chatting with the job database"""
    chatbot = get_chatbot()
    return chatbot.chat(query)

if __name__ == "__main__":
    # Test the chatbot
    test_queries = [
        "What are the latest Engineering jobs?",
        "Show me Science jobs with 1 year experience",
        "Tell me about Photo Type Setter Operator position"
    ]
    
    chatbot = JobYaariRAGChatbot()
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print('='*50)
        
        response = chatbot.chat(query)
        print(f"Answer: {response['answer']}")
        print(f"Sources: {len(response['sources'])} found")
        
        if response['sources']:
            for i, source in enumerate(response['sources'], 1):
                print(f"  {i}. {source['title']} - {source['company']}")