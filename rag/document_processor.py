# rag/document_processor.py
import os
import re
import yaml
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Process raw documents from data/raw into chunks with metadata."""
    
    def __init__(
        self,
        raw_docs_dir: str = "data/raw",
        processed_dir: str = "data/processed",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.raw_docs_dir = raw_docs_dir
        self.processed_dir = processed_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Make sure processed directory exists
        os.makedirs(processed_dir, exist_ok=True)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n## ", "\n### ", "\n\n", "\n", ".", " ", ""],
            length_function=len
        )
    
    def load_document(self, file_path: str) -> Tuple[Dict[str, str], str]:
        """Load a document and separate its YAML metadata and content."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Initialize metadata with filename
        filename = os.path.basename(file_path)
        metadata = {'title': os.path.splitext(filename)[0].replace('-', ' ').title()}
        
        # Extract title from YAML front matter if it exists
        if content.startswith('---'):
            try:
                _, yaml_text, document_text = content.split('---', 2)
                yaml_data = yaml.safe_load(yaml_text.strip())
                if yaml_data and 'title' in yaml_data:
                    metadata['title'] = yaml_data['title']
                return metadata, document_text.strip()
            except (yaml.YAMLError, ValueError) as e:
                logger.warning(f"Error parsing YAML in {file_path}: {e}")
        
        return metadata, content.strip()
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess the document text for ISP documentation."""
        # Replace multiple newlines with a single newline
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove extra whitespace but preserve newlines for formatting
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Clean up technical notation
        text = re.sub(r'(\d+)\s*[mM]bps', r'\1 Mbps', text)  # Standardize Mbps notation
        text = re.sub(r'(\d+)\s*[gG][hH][zZ]', r'\1 GHz', text)  # Standardize GHz notation
        
        # Expand common ISP/tech abbreviations
        isp_terms = {
            r'\bISP\b': 'Internet Service Provider (ISP)',
            r'\bWAN\b': 'Wide Area Network (WAN)',
            r'\bLAN\b': 'Local Area Network (LAN)',
            r'\bWiFi\b': 'Wi-Fi',
            r'\bSSID\b': 'Network Name (SSID)',
            r'\bWPA[23]\b': 'Wi-Fi Protected Access',
            r'\bNAT\b': 'Network Address Translation (NAT)',
            r'\bDHCP\b': 'Dynamic Host Configuration Protocol (DHCP)',
            r'\bDNS\b': 'Domain Name System (DNS)'
        }
        
        for abbr, full in isp_terms.items():
            # Only replace the first occurrence in each document
            text = re.sub(abbr, full, text, count=1)
        
        # Clean up numbered steps
        text = re.sub(r'(\n\s*\d+[.)])\s+', r'\n\n\1 ', text)
        
        return text.strip()
    
    def enhance_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Return only the title from metadata."""
        return {'title': metadata.get('title', 'Untitled Document')}
    

    
    def chunk_document(
        self, 
        text: str, 
        metadata: Dict[str, str]
    ) -> List[Document]:
        """Split document into chunks with title as metadata."""
        # Clean the text first
        cleaned_text = self.clean_text(text)
        
        # Get title from metadata
        title = metadata.get('title', 'Untitled Document')
        
        # Split the text into chunks
        chunks = self.text_splitter.split_text(cleaned_text)
        
        # Create Document objects with metadata
        documents = []
        for i, chunk_text in enumerate(chunks):
            # Add chunk metadata with just the title
            chunk_metadata = {
                'title': title,
                'chunk_id': i,
                'total_chunks': len(chunks)
            }
            
            documents.append(Document(
                page_content=chunk_text,
                metadata=chunk_metadata
            ))
        
        return documents
    
    def process_all_documents(self) -> List[Document]:
        """Process all documents in the raw docs directory."""
        all_documents = []
        
        # Get all markdown files
        for filename in os.listdir(self.raw_docs_dir):
            if filename.endswith('.md'):
                file_path = os.path.join(self.raw_docs_dir, filename)
                logger.info(f"Processing {filename}")
                
                try:
                    # Load and parse the document
                    metadata, content = self.load_document(file_path)
                    
                    # Skip empty documents
                    if not content.strip():
                        logger.warning(f"Empty content in {filename}")
                        continue
                    
                    # Add filename to metadata
                    metadata['filename'] = filename
                    
                    # Process the document
                    document_chunks = self.chunk_document(content, metadata)
                    all_documents.extend(document_chunks)
                    
                    logger.info(f"Created {len(document_chunks)} chunks from {filename}")
                except Exception as e:
                    logger.error(f"Error processing {filename}: {e}")
        
        logger.info(f"Processed {len(all_documents)} total chunks from all documents")
        return all_documents
    
    def save_processed_documents(self, documents: List[Document]):
        """Save processed documents for inspection or later use."""
        import json
        
        # Convert to serializable format
        serializable_docs = []
        for doc in documents:
            serializable_docs.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        
        # Save to JSON
        output_path = os.path.join(self.processed_dir, "processed_chunks.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_docs, f, indent=2, default=str)
        
        logger.info(f"Saved processed documents to {output_path}")

# Example usage
if __name__ == "__main__":
    processor = DocumentProcessor()
    documents = processor.process_all_documents()
    processor.save_processed_documents(documents)
    print(f"Processed {len(documents)} document chunks")