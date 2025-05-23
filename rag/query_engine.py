# rag/query_engine.py
from typing import Dict, List, Any, Optional
import logging
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from .retriever import AdvancedRAGRetriever
from models.llm import MistralLLM

logger = logging.getLogger(__name__)

class ISPQueryEngine:
    """Engine to perform RAG queries about ISP help documentation."""
    
    def __init__(
        self, 
        retriever: AdvancedRAGRetriever
    ):
        self.retriever = retriever
        # Initialize Mistral LLM
        self.llm = MistralLLM()
        
    def _format_sources_for_prompt(self, sources: List[Dict[str, Any]]) -> str:
        """Format retrieved sources into a string for the prompt."""
        formatted_sources = []
        
        for i, source in enumerate(sources, 1):
            formatted_source = f"SOURCE {i}:\nTitle: {source.get('title', 'Untitled Document')}\n\nContent:\n{source.get('content', '')}\n"
            formatted_sources.append(formatted_source)
        
        return "\n".join(formatted_sources)
    
    def generate_formatted_response(
        self, 
        query: str,
        use_query_rewriting: bool = True,
        use_compression: bool = False
    ) -> Dict[str, Any]:
        """Generate a response with source attribution using Mistral API."""
        logger.info(f"Processing query: {query}")
        
        # Get sources based on retrieval method
        if use_compression:
            # Get compressed documents for focused information
            sources = self.retriever.retrieve_with_compression(query)
        else:
            # Get sources with simplified metadata
            sources = self.retriever.retrieve_with_sources(query, rewrite_query=use_query_rewriting)
        
        # Create the sources text for the prompt
        sources_text = self._format_sources_for_prompt(sources)
        
        # Create prompt template
        prompt_template = PromptTemplate(
            input_variables=["query", "sources"],
            template="""You are a helpful and knowledgeable ISP support assistant. Your goal is to provide accurate, 
clear, and helpful responses to customer questions about their internet service.

Using ONLY the information provided in the sources below, provide a helpful response to this query:

CUSTOMER QUESTION: {query}

RELEVANT DOCUMENTATION:
{sources}

INSTRUCTIONS:
1. Provide a clear, step-by-step solution if the question is about troubleshooting
2. Be concise but thorough in your explanations
3. Use simple, non-technical language when possible
4. If referring to specific equipment (modems, routers), mention the model if available
5. If the question is about billing or account-specific information, direct the user to contact customer support
6. Clearly indicate when you're making an educated guess or inference
7. If the sources don't contain enough information, say so and suggest contacting support
8. For technical issues, include basic troubleshooting steps before suggesting a technician visit

HELPFUL RESPONSE:"""
        )
        
        # Generate the response using Mistral API
        chain = prompt_template | self.llm
        response = chain.invoke({
            "query": query,
            "sources": sources_text
        })
        
        # Prepare source attribution for the response
        source_attribution = []
        for i, source in enumerate(sources, 1):
            attribution = {
                "id": i,
                "title": source.get("title", "Untitled Document"),
                "content": source["content"][:500] + "..." if len(source["content"]) > 500 else source["content"]
            }
            source_attribution.append(attribution)
        
        # Return structured response
        result = {
            "query": query,
            "response": response.strip(),
            "sources": source_attribution,
            "rewritten_query": self.retriever.rewrite_query(query) if use_query_rewriting else None
        }
        
        return result