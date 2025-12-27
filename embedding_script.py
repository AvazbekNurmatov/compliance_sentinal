import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from openai import OpenAI
from tqdm import tqdm
import time
from dotenv import load_dotenv

class ChunkEmbeddingGenerator:
    """
    Generate embeddings for chunked banking documents.
    Uses OpenAI's text-embedding-3-small model.
    
    Language-aware embedding:
    - Uzbek documents → embed chunk_text_latin
    - Russian documents → embed chunk_text_cyrillic
    """
    
    def __init__(self, chunks_json_path: str, api_key: Optional[str] = None):
        """
        Initialize with chunks JSON and OpenAI API key.
        
        Args:
            chunks_json_path: Path to bank_chunks_metadata.json
            api_key: OpenAI API key (or load from .env file)
        """
        self.chunks_json_path = Path(chunks_json_path)
        
        # Load environment variables from .env file
        env_path = self.chunks_json_path.parent.parent / '.env'
        if env_path.exists():
            load_dotenv(env_path)
            print(f"✓ Loaded .env from: {env_path}")
        else:
            load_dotenv()  # Try current directory
        
        # Load chunks
        print(f"Loading chunks from: {self.chunks_json_path}")
        with open(self.chunks_json_path, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        
        print(f"✓ Loaded {len(self.chunks)} chunks")
        
        # Initialize OpenAI client
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError(
                "❌ OpenAI API key not found!\n\n"
                "Please create a .env file with:\n"
                "OPENAI_API_KEY=sk-your-api-key-here\n\n"
                "Or set OPENAI_API_KEY environment variable"
            )
        
        self.client = OpenAI(api_key=self.api_key)
        
        # Model configuration
        self.model_name = "text-embedding-3-small"
        self.embedding_dimension = 1536
        self.batch_size = 100
        self.max_retries = 3
        self.retry_delay = 2
        
        print(f"✓ Using model: {self.model_name} ({self.embedding_dimension}D)")
    
    def _get_text_to_embed(self, chunk: Dict) -> tuple[str, str]:
        """
        Get the appropriate text to embed based on document language.
        
        Args:
            chunk: Chunk dictionary with language and text fields
            
        Returns:
            Tuple of (text_to_embed, field_name_used)
        """
        language = chunk.get('document_language', '').strip()
        
        # Determine which field to use based on language
        if language == "Uzbek":
            field_name = 'chunk_text_latin'
            text = chunk.get('chunk_text_latin', '').strip()
        elif language == "Russian" or language == "Русский":
            field_name = 'chunk_text_cyrillic'
            text = chunk.get('chunk_text_cyrillic', '').strip()
        else:
            # Should not happen based on your confirmation, but just in case
            field_name = 'chunk_text_latin'
            text = chunk.get('chunk_text_latin', '').strip()
            print(f"  ⚠️  Warning: Unexpected language '{language}' for chunk {chunk.get('chunk_id')}")
        
        # Validation
        if not text:
            # Try fallback
            fallback_field = 'chunk_text_cyrillic' if field_name == 'chunk_text_latin' else 'chunk_text_latin'
            fallback_text = chunk.get(fallback_field, '').strip()
            
            if fallback_text:
                print(f"  ⚠️  Warning: Empty {field_name} for chunk {chunk.get('chunk_id')}, using {fallback_field}")
                return fallback_text, fallback_field
            else:
                raise ValueError(f"Both text fields are empty for chunk {chunk.get('chunk_id')}")
        
        return text, field_name
    
    def generate_embeddings(self, output_dir: str = 'embeddings', skip_confirmation: bool = False):
        """
        Generate embeddings for all chunks.
        
        Args:
            output_dir: Directory to save embeddings
            skip_confirmation: Skip confirmation prompt (useful for automation)
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Analyze chunks by language
        language_stats = {}
        script_stats = {}
        for chunk in self.chunks:
            lang = chunk.get('document_language', 'unknown')
            script = chunk.get('document_script', 'unknown')
            language_stats[lang] = language_stats.get(lang, 0) + 1
            script_stats[script] = script_stats.get(script, 0) + 1
        
        print("\n" + "="*80)
        print("EMBEDDING GENERATION")
        print("="*80)
        print(f"Model: {self.model_name}")
        print(f"Embedding dimension: {self.embedding_dimension}")
        print(f"Chunks to process: {len(self.chunks)}")
        print(f"\nLanguage distribution:")
        for lang, count in sorted(language_stats.items()):
            print(f"  • {lang}: {count} chunks")
        print(f"\nScript distribution:")
        for script, count in sorted(script_stats.items()):
            print(f"  • {script}: {count} chunks")
        print(f"\nEmbedding strategy:")
        print(f"  • Uzbek documents → chunk_text_latin")
        print(f"  • Russian documents → chunk_text_cyrillic")
        print(f"\nBatch size: {self.batch_size}")
        print(f"Output directory: {output_path}")
        print("="*80)
        
        if not skip_confirmation:
            proceed = input("\n⚠️  Proceed? This will use OpenAI API credits. (yes/no): ").strip().lower()
            if proceed != 'yes':
                print("❌ Cancelled.")
                return None
        
        # Process chunks in batches
        all_embeddings = []
        failed_chunks = []
        
        total_batches = (len(self.chunks) + self.batch_size - 1) // self.batch_size
        
        print(f"\n🚀 Processing {total_batches} batches...\n")
        
        for batch_idx in range(0, len(self.chunks), self.batch_size):
            batch_chunks = self.chunks[batch_idx:batch_idx + self.batch_size]
            batch_num = (batch_idx // self.batch_size) + 1
            
            print(f"[Batch {batch_num}/{total_batches}] Processing chunks {batch_idx} to {batch_idx + len(batch_chunks) - 1}...")
            
            try:
                batch_embeddings = self._embed_batch(batch_chunks)
                all_embeddings.extend(batch_embeddings)
                print(f"  ✓ Successfully embedded {len(batch_embeddings)} chunks")
            except Exception as e:
                print(f"  ✗ Batch failed: {str(e)}")
                failed_chunks.extend(batch_chunks)
            
            # Rate limiting - small delay between batches
            if batch_idx + self.batch_size < len(self.chunks):
                time.sleep(0.5)
        
        # Save embeddings
        print(f"\n{'='*80}")
        print("💾 SAVING RESULTS")
        print("="*80)
        
        self._save_embeddings(all_embeddings, output_path)
        
        if failed_chunks:
            self._save_failed_chunks(failed_chunks, output_path)
            print(f"\n⚠️  {len(failed_chunks)} chunks failed. See failed_chunks.json")
        
        print(f"\n✅ Successfully generated {len(all_embeddings)}/{len(self.chunks)} embeddings")
        print(f"✅ Saved to: {output_path}")
        
        return all_embeddings
    
    def _embed_batch(self, batch_chunks: List[Dict]) -> List[Dict]:
        """
        Embed a batch of chunks with retry logic.
        Uses appropriate text based on document language.
        """
        # Get appropriate text for each chunk based on language
        texts_and_fields = [self._get_text_to_embed(chunk) for chunk in batch_chunks]
        texts = [t[0] for t in texts_and_fields]
        field_names = [t[1] for t in texts_and_fields]
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=texts
                )
                
                # Prepare embedding records
                embeddings = []
                for i, chunk in enumerate(batch_chunks):
                    embedding_vector = response.data[i].embedding
                    
                    embeddings.append({
                        'chunk_id': chunk['chunk_id'],
                        'document_filename': chunk['document_filename'],
                        'document_date': chunk.get('document_date'),
                        'document_status': chunk.get('document_status'),
                        'document_language': chunk['document_language'],
                        'document_script': chunk.get('document_script'),
                        'document_folder': chunk.get('document_folder'),
                        'section_number': chunk.get('section_number'),
                        'chunk_index': chunk['chunk_index'],
                        'total_chunks': chunk.get('total_chunks'),
                        'chunk_position': chunk.get('chunk_position'),
                        'chunk_text_latin': chunk.get('chunk_text_latin', ''),
                        'chunk_text_cyrillic': chunk.get('chunk_text_cyrillic', ''),
                        'chunk_length': chunk.get('chunk_length'),
                        'chunk_words': chunk['chunk_words'],
                        'embedded_text_field': field_names[i],
                        'embedding': embedding_vector,
                        'embedding_model': self.model_name,
                        'embedding_dimension': len(embedding_vector),
                        'compliance_flags': chunk.get('compliance_flags', []),
                        'ready_for_embedding': chunk.get('ready_for_embedding', True)
                    })
                
                return embeddings
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"    ⚠️  Attempt {attempt + 1} failed: {str(e)}. Retrying in {self.retry_delay}s...")
                    time.sleep(self.retry_delay)
                else:
                    raise
    
    def _save_embeddings(self, embeddings: List[Dict], output_path: Path):
        """Save embeddings in multiple formats."""
        
        if not embeddings:
            print("⚠️  No embeddings to save")
            return
        
        # 1. Save full JSON with embeddings
        json_file = output_path / 'chunk_embeddings.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(embeddings, f, ensure_ascii=False, indent=2)
        file_size_mb = json_file.stat().st_size / (1024 * 1024)
        print(f"✓ Saved full embeddings: {json_file} ({file_size_mb:.2f} MB)")
        
        # 2. Save embeddings as numpy array (for efficient loading)
        embedding_vectors = np.array([e['embedding'] for e in embeddings])
        npy_file = output_path / 'embeddings.npy'
        np.save(npy_file, embedding_vectors)
        npy_size_mb = npy_file.stat().st_size / (1024 * 1024)
        print(f"✓ Saved numpy array: {npy_file} (shape: {embedding_vectors.shape}, {npy_size_mb:.2f} MB)")
        
        # 3. Save metadata index (without embeddings for easy viewing)
        metadata_records = []
        for e in embeddings:
            meta = {k: v for k, v in e.items() if k != 'embedding'}
            metadata_records.append(meta)
        
        metadata_df = pd.DataFrame(metadata_records)
        csv_file = output_path / 'embeddings_index.csv'
        metadata_df.to_csv(csv_file, index=False, encoding='utf-8')
        csv_size_mb = csv_file.stat().st_size / (1024 * 1024)
        print(f"✓ Saved metadata index: {csv_file} ({len(metadata_df)} rows, {csv_size_mb:.2f} MB)")
        
        # 4. Save summary statistics
        stats = {
            'generation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_embeddings': len(embeddings),
            'embedding_model': self.model_name,
            'embedding_dimension': self.embedding_dimension,
            'documents_processed': len(set(e['document_filename'] for e in embeddings)),
            'language_distribution': {},
            'script_distribution': {},
            'embedded_text_field_distribution': {},
            'document_status_distribution': {},
            'avg_chunk_words': sum(e['chunk_words'] for e in embeddings) / len(embeddings) if embeddings else 0,
            'avg_chunk_length': sum(e.get('chunk_length', 0) for e in embeddings) / len(embeddings) if embeddings else 0,
            'compliance_distribution': {},
            'unique_document_folders': list(set(e.get('document_folder', 'unknown') for e in embeddings))
        }
        
        for e in embeddings:
            # Language stats
            lang = e['document_language']
            stats['language_distribution'][lang] = stats['language_distribution'].get(lang, 0) + 1
            
            # Script stats
            script = e.get('document_script', 'unknown')
            stats['script_distribution'][script] = stats['script_distribution'].get(script, 0) + 1
            
            # Embedded field stats
            embedded_field = e.get('embedded_text_field', 'unknown')
            stats['embedded_text_field_distribution'][embedded_field] = (
                stats['embedded_text_field_distribution'].get(embedded_field, 0) + 1
            )
            
            # Document status stats
            status = e.get('document_status', 'unknown')
            stats['document_status_distribution'][status] = stats['document_status_distribution'].get(status, 0) + 1
            
            # Compliance flags
            for flag in e['compliance_flags']:
                stats['compliance_distribution'][flag] = stats['compliance_distribution'].get(flag, 0) + 1
        
        stats_file = output_path / 'embedding_statistics.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"✓ Saved statistics: {stats_file}")
        
        print(f"\n📊 STATISTICS SUMMARY:")
        print(f"{'─'*80}")
        print(f"  Total embeddings: {stats['total_embeddings']:,}")
        print(f"  Unique documents: {stats['documents_processed']}")
        print(f"  Average words/chunk: {stats['avg_chunk_words']:.1f}")
        print(f"  Average characters/chunk: {stats['avg_chunk_length']:.1f}")
        print(f"\n  Language distribution:")
        for lang, count in sorted(stats['language_distribution'].items()):
            percentage = (count / stats['total_embeddings']) * 100
            print(f"    • {lang}: {count:,} ({percentage:.1f}%)")
        print(f"\n  Embedded text field distribution:")
        for field, count in sorted(stats['embedded_text_field_distribution'].items()):
            percentage = (count / stats['total_embeddings']) * 100
            print(f"    • {field}: {count:,} ({percentage:.1f}%)")
        if stats['compliance_distribution']:
            print(f"\n  Compliance flags found:")
            for flag, count in sorted(stats['compliance_distribution'].items()):
                print(f"    • {flag}: {count}")
        print(f"{'─'*80}")
    
    def _save_failed_chunks(self, failed_chunks: List[Dict], output_path: Path):
        """Save chunks that failed to embed."""
        failed_file = output_path / 'failed_chunks.json'
        with open(failed_file, 'w', encoding='utf-8') as f:
            json.dump(failed_chunks, f, ensure_ascii=False, indent=2)
        print(f"✓ Saved failed chunks: {failed_file} ({len(failed_chunks)} chunks)")


def main():
    """Main execution function."""
    script_dir = Path(__file__).resolve().parent
    chunks_path = script_dir / 'chunked_output' / 'bank_chunks_metadata.json'
    
    print("="*80)
    print("CHUNK EMBEDDING GENERATOR")
    print("="*80)
    
    if not chunks_path.exists():
        print(f"❌ ERROR: Chunks file not found at {chunks_path}")
        print("Please run chunking_script.py first.")
        exit(1)
    
    print(f"✓ Found chunks file: {chunks_path}")
    
    try:
        # Initialize generator (will load API key from .env)
        generator = ChunkEmbeddingGenerator(
            chunks_json_path=str(chunks_path)
        )
        
        # Generate embeddings
        output_dir = script_dir / 'embeddings'
        embeddings = generator.generate_embeddings(output_dir=str(output_dir))
        
        if embeddings:
            print("\n" + "="*80)
            print("✅ COMPLETE - ALL EMBEDDINGS GENERATED SUCCESSFULLY!")
            print("="*80)
            print(f"\nYou can now use the embeddings for:")
            print(f"  • Semantic search")
            print(f"  • Document similarity")
            print(f"  • Compliance checking")
            print(f"  • RAG (Retrieval Augmented Generation)")
        
    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}")
        exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
