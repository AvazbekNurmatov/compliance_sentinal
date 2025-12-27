"""
PDF Processor for Compliance Checking
Extracts text, chunks it, generates embeddings, and stores in ChromaDB
"""

import os
import re
from typing import List, Dict, Tuple
from pathlib import Path
from datetime import datetime

# PDF extraction
import fitz  # PyMuPDF

# OpenAI for embeddings
from openai import OpenAI
from dotenv import load_dotenv

# ChromaDB
import chromadb

# Load environment variables
load_dotenv()

class PDFProcessor:
    """Process PDF files for compliance checking"""
    
    def __init__(self, chroma_path: str = "./index/chroma_db"):
        """Initialize with OpenAI client and ChromaDB"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")
        
        self.client = OpenAI(api_key=api_key)
        self.embedding_model = "text-embedding-3-small"
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        
        # Get or create collection for uploaded documents
        self.uploaded_collection = self.chroma_client.get_or_create_collection(
            name="uploaded_documents"
        )
        
        print(f"✅ Connected to ChromaDB at: {chroma_path}")
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract all text from PDF file
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text as string
        """
        doc = None
        try:
            # Open PDF document
            doc = fitz.open(pdf_path)
            
            # Check if document opened successfully
            if doc is None or doc.is_closed:
                raise Exception("Failed to open PDF document")
            
            page_count = doc.page_count
            print(f"📖 PDF has {page_count} pages")
            
            full_text = ""
            
            # Extract text from each page
            for page_num in range(page_count):
                try:
                    page = doc.load_page(page_num)
                    text = page.get_text("text")
                    
                    if text.strip():
                        full_text += text + "\n"
                    
                    print(f"  ✓ Extracted page {page_num + 1}/{page_count}")
                    
                except Exception as page_error:
                    print(f"  ⚠️  Warning: Could not extract page {page_num + 1}: {page_error}")
                    continue
            
            print(f"✅ Extracted {len(full_text)} characters from {page_count} pages")
            return full_text.strip()
            
        except Exception as e:
            raise Exception(f"Error extracting PDF text: {str(e)}")
            
        finally:
            # Always close the document
            if doc is not None and not doc.is_closed:
                doc.close()
    
    def clean_text(self, text: str) -> str:
        """
        Clean extracted text
        
        Args:
            text: Raw text from PDF
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace but preserve line breaks
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Remove extra spaces within line
            line = re.sub(r'\s+', ' ', line)
            line = line.strip()
            if line:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def chunk_text(self, text: str, min_words: int = 10) -> List[Dict]:
        """
        Split text into chunks using paragraph-based logic
        Rules:
        - New line starting with capital letter = new chunk
        - New line starting with number (e.g., "1.", "7.4.5.") = new chunk
        - Line ending with . or ; = new chunk after it
        - New line with 2+ leading spaces = new chunk
        
        Args:
            text: Full text to chunk
            min_words: Minimum words required for a valid chunk
            
        Returns:
            List of chunk dictionaries
        """
        # Clean text first
        text = self.clean_text(text)
        
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        
        # Pattern for section numbers (e.g., "1.", "1.2.", "7.4.5.3.")
        section_pattern = re.compile(r'^\s*\d+\.(?:\d+\.)*')
        
        for line in lines:
            line_stripped = line.strip()
            
            # Skip empty lines
            if not line_stripped:
                continue
            
            # Check if line starts with section number
            starts_with_number = bool(section_pattern.match(line_stripped))
            
            # Check if line starts with capital letter (Cyrillic or Latin)
            starts_with_capital = bool(re.match(r'^[А-ЯЁA-ZĞŞÇÖÜIİ]', line_stripped))
            
            # Check if line starts with 2+ spaces
            starts_with_spaces = len(line) - len(line.lstrip()) >= 2
            
            # Check if previous line ended with . or ;
            ends_with_period_or_semicolon = False
            if current_chunk:
                last_line = current_chunk[-1].rstrip()
                ends_with_period_or_semicolon = last_line.endswith('.') or last_line.endswith(';')
            
            # New chunk conditions
            should_split = ((starts_with_number or starts_with_capital or starts_with_spaces or ends_with_period_or_semicolon) 
                           and current_chunk)
            
            if should_split:
                # Save current chunk
                chunk_text = ' '.join(current_chunk).strip()
                if len(chunk_text.split()) >= min_words:
                    chunks.append(chunk_text)
                current_chunk = []
            
            # Add line to current chunk
            current_chunk.append(line_stripped)
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk).strip()
            if len(chunk_text.split()) >= min_words:
                chunks.append(chunk_text)
        
        # Create chunk objects with metadata
        chunk_objects = []
        for i, chunk_text in enumerate(chunks):
            chunk_data = {
                "chunk_id": f"uploaded_chunk_{i}",
                "chunk_index": i,
                "chunk_text": chunk_text,
                "word_count": len(chunk_text.split()),
                "chunk_position": f"{i + 1}/{len(chunks)}",
                "total_chunks": len(chunks)
            }
            chunk_objects.append(chunk_data)
        
        print(f"✅ Created {len(chunk_objects)} chunks from text (paragraph-based)")
        return chunk_objects
    
    def generate_embeddings(self, chunks: List[Dict]) -> List[Dict]:
        """
        Generate embeddings for chunks using OpenAI
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Chunks with embeddings added
        """
        texts_to_embed = [chunk["chunk_text"] for chunk in chunks]
        
        try:
            print(f"🔄 Generating embeddings for {len(texts_to_embed)} chunks...")
            
            # Call OpenAI API (same model as your existing setup)
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=texts_to_embed
            )
            
            # Add embeddings to chunks
            for i, chunk in enumerate(chunks):
                chunk["embedding"] = response.data[i].embedding
            
            print(f"✅ Generated {len(chunks)} embeddings using {self.embedding_model}")
            return chunks
            
        except Exception as e:
            raise Exception(f"Error generating embeddings: {str(e)}")
    
    def store_in_chromadb(self, chunks: List[Dict], pdf_filename: str) -> str:
        """
        Store chunks with embeddings in ChromaDB
        
        Args:
            chunks: Chunks with embeddings
            pdf_filename: Name of source PDF file
            
        Returns:
            Document ID for retrieval
        """
        try:
            print(f"💾 Storing {len(chunks)} chunks in ChromaDB...")
            
            # Prepare data for ChromaDB
            ids = [chunk["chunk_id"] for chunk in chunks]
            embeddings = [chunk["embedding"] for chunk in chunks]
            documents = [chunk["chunk_text"] for chunk in chunks]
            
            # Prepare metadata (remove embedding to avoid duplication)
            metadatas = []
            for chunk in chunks:
                metadata = {
                    "source_pdf": pdf_filename,
                    "chunk_index": chunk["chunk_index"],
                    "word_count": chunk["word_count"],
                    "chunk_position": chunk["chunk_position"],
                    "upload_date": datetime.now().isoformat()
                }
                metadatas.append(metadata)
            
            # Add to ChromaDB collection
            self.uploaded_collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            doc_id = f"uploaded_{pdf_filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            print(f"✅ Stored in ChromaDB collection: uploaded_documents")
            print(f"   Document ID: {doc_id}")
            
            return doc_id
            
        except Exception as e:
            raise Exception(f"Error storing in ChromaDB: {str(e)}")
    
    def process_pdf(self, pdf_path: str) -> Tuple[List[Dict], str, str]:
        """
        Complete PDF processing pipeline
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Tuple of (chunks_with_embeddings, full_text, document_id)
        """
        print(f"\n{'='*60}")
        print(f"📄 Processing PDF: {pdf_path}")
        print(f"{'='*60}\n")
        
        pdf_filename = Path(pdf_path).name
        
        # Step 1: Extract text
        full_text = self.extract_text_from_pdf(pdf_path)
        
        # Step 2: Chunk text
        chunks = self.chunk_text(full_text)
        
        # Step 3: Generate embeddings
        chunks_with_embeddings = self.generate_embeddings(chunks)
        
        # Step 4: Store in ChromaDB
        doc_id = self.store_in_chromadb(chunks_with_embeddings, pdf_filename)
        
        print(f"\n{'='*60}")
        print(f"✅ PDF Processing Complete!")
        print(f"{'='*60}\n")
        
        return chunks_with_embeddings, full_text, doc_id


def print_sample_chunks(chunks: List[Dict], n: int = 3):
    """Print sample chunks for verification"""
    print(f"\n📋 Sample Chunks (showing first {n}):\n")
    
    for i in range(min(n, len(chunks))):
        chunk = chunks[i]
        print(f"--- Chunk {i + 1} ---")
        print(f"ID: {chunk['chunk_id']}")
        print(f"Words: {chunk['word_count']}")
        print(f"Position: {chunk['chunk_position']}")
        print(f"Text Preview: {chunk['chunk_text'][:200]}...")
        print(f"Embedding dimension: {len(chunk['embedding'])}")
        print()


if __name__ == "__main__":
    """Test the PDF processor"""
    
    # Initialize processor
    processor = PDFProcessor()
    
    # Test with sample PDF
    pdf_file = "sample_bank_paper.pdf"
    
    # Check if file exists
    if not os.path.exists(pdf_file):
        print(f"❌ Error: {pdf_file} not found in current directory")
        print(f"Current directory: {os.getcwd()}")
        print(f"Please place your PDF in: {os.path.abspath('.')}")
        exit(1)
    
    try:
        # Process the PDF
        chunks_with_embeddings, full_text, doc_id = processor.process_pdf(pdf_file)
        
        # Print summary
        print(f"📊 Processing Summary:")
        print(f"  - Total text length: {len(full_text)} characters")
        print(f"  - Total chunks: {len(chunks_with_embeddings)}")
        print(f"  - Embedding model: {processor.embedding_model}")
        print(f"  - Embedding dimension: {len(chunks_with_embeddings[0]['embedding'])}")
        print(f"  - ChromaDB Document ID: {doc_id}")
        
        # Show sample chunks
        print_sample_chunks(chunks_with_embeddings, n=3)
        
        # Show ChromaDB info
        print(f"\n💾 ChromaDB Collection Info:")
        print(f"  - Collection name: uploaded_documents")
        print(f"  - Total items: {processor.uploaded_collection.count()}")
        
        print("\n✨ Test completed successfully!")
        print("\n💡 Next step: Run compliance_checker.py to compare against regulations and bank_policies")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        exit(1)
