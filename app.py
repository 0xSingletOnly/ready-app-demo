# app.py
import streamlit as st
import os
from rag.document_processor import DocumentProcessor
from rag.retriever import AdvancedRAGRetriever
from rag.query_engine import ISPQueryEngine
from models.llm import MistralLLM

# Set page configuration
st.set_page_config(
    page_title="Ready Now - ISP Help Center",
    page_icon="❓",
    layout="wide"
)

# Function to generate base model response without RAG
def generate_base_response(query):
    """Generate a response using only the base Mistral model (no RAG)."""
    base_llm = MistralLLM()
    prompt = f"""You are a helpful ISP support assistant. Answer this question about internet service, 
modems, routers, connectivity issues, billing, and other ISP-related topics:

QUESTION: {query}

Provide a clear, helpful response based on your knowledge of common ISP issues and solutions.
If you don't know the answer, say so rather than making up information."""
    
    response = base_llm.invoke(prompt)
    return response

# Initialize session state for storing the query engine
if 'query_engine' not in st.session_state:
    with st.spinner("Initializing the ISP Help System..."):
        # Process documents with larger chunks for better context
        processor = DocumentProcessor(chunk_size=4000, chunk_overlap=800)
        documents = processor.process_all_documents()
        
        # Initialize retriever with ISP-specific settings
        retriever = AdvancedRAGRetriever(
            documents=documents,
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
            persist_directory="db/chroma",
            k=3  # Retrieve 3 most relevant documents
        )
        
        # Initialize query engine
        st.session_state.query_engine = ISPQueryEngine(retriever=retriever)
    st.success("Help system ready! Ask me anything about your ISP services.")

# Title and introduction
st.title("❓ ISP Help Center")
st.markdown("""
Welcome to your ISP's AI-powered help center. Ask me anything about your internet service, 
such as troubleshooting, setup instructions, billing questions, or service details.
""")

# Query input
default_question = "How do I change my Xfinity appointment?"
query = st.text_input("Ask me anything about your ISP service:", 
                     default_question)

# Options
with st.expander("Search Options"):
    col1, col2 = st.columns(2)
    with col1:
        use_query_rewriting = st.checkbox("Expand Search Terms", value=True, 
                                        help="Include related technical terms in search")
    with col2:
        show_sources = st.checkbox("Show Source Documents", value=True,
                                 help="Display the documentation used to generate the answer")

# Process the query
if st.button("Get Help"):
    if not query:
        st.warning("Please enter your question about our services.")
    else:
        with st.spinner("Searching our knowledge base..."):
            # Generate response using the query engine
            result = st.session_state.query_engine.generate_formatted_response(
                query=query,
                use_query_rewriting=use_query_rewriting
            )
            
            # Display the main response
            st.markdown("### Here's what we found:")
            st.markdown(result["response"])
            
            # Show query interpretation if available
            if use_query_rewriting and result.get("rewritten_query"):
                with st.expander("How I interpreted your question"):
                    st.info(result["rewritten_query"])
            
            # Show sources if enabled
            if show_sources and "sources" in result and result["sources"]:
                st.markdown("---")
                st.markdown("### Sources Used")
                for source in result["sources"]:
                    with st.expander(source.get('title', 'Document')):
                        st.markdown("**Relevant information:**")
                        st.text(source.get("content", "No content available"))
