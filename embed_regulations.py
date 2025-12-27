import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from openai import OpenAI
import time
from dotenv import load_dotenv

class RegulationEmbeddingGenerator:
    """
    Generate embeddings for regulation documents.
    Specifically targets 'chunk_text_latin' regardless of source script.
    """
    
    def __init__(self, chunks_json_path: str, api_key: Optional[str] = None):
        self.chunks_json_path = Path(chunks_json_path)
        
        # Load .env from project root
        load_dotenv()
        
        # Load chunks
        print(f"Reading: {self.chunks_json_path}")
        with open(self.chunks_json_path, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        
        print(f"✓ Loaded {len(self.chunks)} regulation chunks")
        
        # Initialize OpenAI
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("❌ OpenAI API key not found in .env or environment.")
        
        self.client = OpenAI(api_key=self.api_key)
        
        # Config
        self.model_name = "text-embedding-3-small"
        self.embedding_dimension = 1536
        self.batch_size = 100
        self.max_retries = 3

    def generate_embeddings(self, output_dir: str):
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        print("\n" + "="*50)
        print(f"Targeting Field: chunk_text_latin")
        print(f"Output Directory: {output_path}")
        print("="*50)

        all_embeddings = []
        total_batches = (len(self.chunks) + self.batch_size - 1) // self.batch_size
        
        for i in range(0, len(self.chunks), self.batch_size):
            batch_chunks = self.chunks[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            
            print(f"Processing Batch {batch_num}/{total_batches}...")
            
            try:
                # Always take chunk_text_latin as requested
                texts = [chunk.get('chunk_text_latin', '') for chunk in batch_chunks]
                
                # API Call with Retries
                response = self._fetch_embeddings_with_retry(texts)
                
                # Build rich records
                for j, chunk in enumerate(batch_chunks):
                    record = {
                        **chunk,  # Keep all original fields (id, filename, index, cyrillic, metadata)
                        'embedding': response.data[j].embedding,
                        'embedding_model': self.model_name
                    }
                    all_embeddings.append(record)
                    
            except Exception as e:
                print(f"❌ Critical failure at batch {batch_num}: {e}")
                continue

        self._save_results(all_embeddings, output_path)

    def _fetch_embeddings_with_retry(self, texts: List[str]):
        for attempt in range(self.max_retries):
            try:
                return self.client.embeddings.create(model=self.model_name, input=texts)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2)
                else:
                    raise e

    def _save_results(self, embeddings: List[Dict], output_path: Path):
        # 1. Save Full JSON
        json_file = output_path / 'regulation_embeddings.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(embeddings, f, ensure_ascii=False, indent=2)
        
        # 2. Save Weights (.npy)
        vectors = np.array([e['embedding'] for e in embeddings])
        np.save(output_path / 'regulation_vectors.npy', vectors)
        
        # 3. Save Flat Metadata (CSV)
        meta_only = []
        for e in embeddings:
            row = {k: v for k, v in e.items() if k != 'embedding'}
            # Flatten metadata dict if it exists
            if 'metadata' in row:
                for mk, mv in row['metadata'].items():
                    row[f'meta_{mk}'] = mv
                del row['metadata']
            meta_only.append(row)
            
        pd.DataFrame(meta_only).to_csv(output_path / 'regulation_index.csv', index=False)
        print(f"\n✅ SUCCESS: Saved to {output_path}")

def main():
    # Strict pathing based on your tree
    base_dir = Path(__file__).resolve().parent
    chunks_path = base_dir / 'chunked_output' / 'chunked_regulations' / 'regulation_chunks_metadata.json'
    output_dir = base_dir / 'embeddings' / 'regulation_embeddings'
    
    if not chunks_path.exists():
        print(f"❌ File not found: {chunks_path}")
        return

    generator = RegulationEmbeddingGenerator(chunks_json_path=str(chunks_path))
    generator.generate_embeddings(output_dir=str(output_dir))

if __name__ == "__main__":
    main()
