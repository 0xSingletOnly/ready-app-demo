# scripts/ingest_documents.py
import sys
import os

# Add the project root to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag.document_processor import DocumentProcessor

def main():
    processor = DocumentProcessor(
        raw_docs_dir="data/raw",
        processed_dir="data/processed",
        chunk_size=8000,  # Adjust based on your model's context window
        chunk_overlap=1600
    )
    
    documents = processor.process_all_documents()
    processor.save_processed_documents(documents)
    
    print(f"Successfully processed {len(documents)} document chunks")
    
    return documents

if __name__ == "__main__":
    main()