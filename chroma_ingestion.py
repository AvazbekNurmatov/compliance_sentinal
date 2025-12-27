"""
Re-ingest ChromaDB with documents field populated
This fixes the issue where text was only in metadata, not in searchable documents
"""

import chromadb
import numpy as np
import pandas as pd
import os

# Paths
CHROMA_DATA_PATH = "./index/chroma_db"
BANK_VECTORS = "embeddings/bank_policy_embeddings/embeddings.npy"
BANK_METADATA = "embeddings/bank_policy_embeddings/embeddings_index.csv"
REG_VECTORS = "embeddings/regulation_embeddings/regulation_vectors.npy"
REG_METADATA = "embeddings/regulation_embeddings/regulation_index.csv"

# Initialize Chroma Client
client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)

def reingest_with_documents(collection_name, npy_path, csv_path, text_field='chunk_text_latin'):
    """
    Re-ingest collection with proper documents field
    
    Args:
        collection_name: Name of ChromaDB collection
        npy_path: Path to embeddings .npy file
        csv_path: Path to metadata CSV
        text_field: Which field contains the text to use as document
    """
    print(f"\n{'='*60}")
    print(f"📦 Re-ingesting: {collection_name}")
    print(f"{'='*60}\n")
    
    # Delete old collection
    try:
        client.delete_collection(name=collection_name)
        print(f"✓ Deleted old {collection_name} collection")
    except:
        print(f"  (No old collection to delete)")
    
    # Create fresh collection
    collection = client.create_collection(name=collection_name)
    
    # Load data
    print(f"📂 Loading embeddings from {npy_path}...")
    vectors = np.load(npy_path).astype('float32').tolist()
    
    print(f"📂 Loading metadata from {csv_path}...")
    metadata_df = pd.read_csv(csv_path)
    
    print(f"   Found {len(vectors)} embeddings and {len(metadata_df)} metadata rows")
    
    # Ensure lengths match
    if len(vectors) != len(metadata_df):
        raise ValueError(f"Mismatch: {len(vectors)} vectors vs {len(metadata_df)} metadata rows!")
    
    # Extract document texts from metadata
    print(f"📝 Extracting document texts from '{text_field}' field...")
    
    # Check which text field exists
    if text_field in metadata_df.columns:
        documents = metadata_df[text_field].fillna("").tolist()
    elif 'chunk_text_cyrillic' in metadata_df.columns:
        print(f"   ⚠️  '{text_field}' not found, using 'chunk_text_cyrillic' instead")
        documents = metadata_df['chunk_text_cyrillic'].fillna("").tolist()
    else:
        raise ValueError(f"No text field found! Available columns: {metadata_df.columns.tolist()}")
    
    # Generate IDs
    ids = [f"id_{i}" for i in range(len(vectors))]
    
    # Convert metadata to dict (keep original metadata)
    metadatas = metadata_df.to_dict('records')
    
    # Batch upload with documents field
    batch_size = 1000
    total_batches = (len(ids) + batch_size - 1) // batch_size
    
    print(f"\n💾 Uploading in {total_batches} batches of {batch_size}...")
    
    for batch_num, i in enumerate(range(0, len(ids), batch_size), 1):
        end = min(i + batch_size, len(ids))
        
        collection.add(
            ids=ids[i:end],
            embeddings=vectors[i:end],
            documents=documents[i:end],  # ← THIS WAS MISSING!
            metadatas=metadatas[i:end]
        )
        
        print(f"  ✓ Batch {batch_num}/{total_batches} ({end}/{len(ids)} items)", end='\r')
    
    print(f"\n\n✅ Done! {collection.count()} items now in {collection_name}")
    
    # Verify
    print(f"\n🔍 Verifying...")
    sample = collection.get(limit=1, include=["documents", "metadatas"])
    
    if sample['documents'] and sample['documents'][0]:
        print(f"✓ Documents field populated correctly!")
        print(f"  Sample text: {sample['documents'][0][:100]}...")
    else:
        print(f"❌ ERROR: Documents field still NULL!")
    
    return collection


def main():
    print(f"\n{'='*60}")
    print("🔄 CHROMADB RE-INGESTION WITH DOCUMENTS FIELD")
    print(f"{'='*60}\n")
    
    print("⚠️  WARNING: This will delete and recreate collections!")
    print("   Make sure you have backups of:")
    print(f"   - {BANK_VECTORS}")
    print(f"   - {BANK_METADATA}")
    print(f"   - {REG_VECTORS}")
    print(f"   - {REG_METADATA}")
    
    proceed = input("\n   Type 'yes' to proceed: ").strip().lower()
    
    if proceed != 'yes':
        print("\n❌ Aborted")
        return
    
    # Re-ingest bank policies
    reingest_with_documents(
        collection_name="bank_policies",
        npy_path=BANK_VECTORS,
        csv_path=BANK_METADATA,
        text_field='chunk_text_latin'  # or 'chunk_text_cyrillic'
    )
    
    # Re-ingest regulations
    reingest_with_documents(
        collection_name="regulations",
        npy_path=REG_VECTORS,
        csv_path=REG_METADATA,
        text_field='chunk_text_latin'  # or 'chunk_text_cyrillic'
    )
    
    print(f"\n{'='*60}")
    print("✅ RE-INGESTION COMPLETE!")
    print(f"{'='*60}\n")
    print("Now run: python3 compliance_checker.py")


if __name__ == "__main__":
    main()
