import chromadb
import numpy as np

client = chromadb.PersistentClient(path="./index/chroma_db")

# Check each collection
for col_name in ["regulations", "bank_policies", "uploaded_documents"]:
    try:
        col = client.get_collection(col_name)
        print(f"\n{'='*60}")
        print(f"📦 Collection: {col_name}")
        print(f"{'='*60}")
        print(f"Total items: {col.count()}")
        
        # Get a sample item
        sample = col.get(limit=1, include=["embeddings", "documents", "metadatas"])
        
        print(f"\n✓ Has embeddings: {sample['embeddings'] is not None and len(sample['embeddings']) > 0}")
        print(f"✓ Has documents: {sample['documents'] is not None and len(sample['documents']) > 0}")
        print(f"✓ Has metadata: {sample['metadatas'] is not None and len(sample['metadatas']) > 0}")
        
        # Check document content
        if sample['documents'] and len(sample['documents']) > 0:
            doc = sample['documents'][0]
            if doc is None:
                print(f"\n❌ PROBLEM: Documents field is NULL!")
            else:
                print(f"\n✓ Sample document text (first 150 chars):")
                print(f"   {doc[:150]}...")
        else:
            print(f"\n❌ PROBLEM: No documents returned!")
            
        # Check metadata
        if sample['metadatas'] and len(sample['metadatas']) > 0:
            print(f"\n✓ Metadata keys: {list(sample['metadatas'][0].keys())}")
        
        # Check embedding dimension
        if sample['embeddings'] and len(sample['embeddings']) > 0:
            emb = sample['embeddings'][0]
            print(f"\n✓ Embedding dimension: {len(emb)}")
            
    except Exception as e:
        print(f"\n❌ Error with {col_name}: {e}")
        import traceback
        traceback.print_exc()
