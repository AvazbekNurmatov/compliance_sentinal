import os
import json
import pandas as pd
from pathlib import Path
from PyPDF2 import PdfReader
from typing import List, Dict
import re

class ParagraphDocumentChunker:
    """
    Strict paragraph-based chunking for Uzbek/Russian banking documents.
    Rules:
    - New line with sentence start (capital letter) = new chunk
    - Line with 2+ leading spaces = new chunk
    - Section headers are NOT included in chunk text
    - Stores both Latin and Cyrillic versions
    """
    
    def __init__(self, metadata_csv_path: str):
        """Initialize with metadata CSV."""
        self.metadata_df = pd.read_csv(metadata_csv_path)
        
        if 'Full_Path' not in self.metadata_df.columns:
            script_dir = Path(__file__).resolve().parent
            self.metadata_df['Full_Path'] = self.metadata_df.apply(
                lambda row: str(script_dir / 'bank_policies' / 'anorbank' / row['Folder'] / row['Filename']),
                axis=1
            )
        
        self.chunks_output = []
        
        self.compliance_keywords = {
            'interest_rate': ['процентная ставка', 'foiz stavkasi', 'фоиз ставкаси'],
            'collateral': ['залог', 'garov', 'гаров'],
            'mandatory_disclosure': ['клиент имеет право', 'mijoz huquqi', 'мижоз ҳуқуқи'],
            'pre_selected': ['по умолчанию', 'avtomatik tanlangan', 'автоматик танланган', '☑', '✓'],
            'language_requirement': ['государственный язык', 'davlat tili', 'давлат тили']
        }
        
        # Latin to Cyrillic mapping for Uzbek
        self.latin_to_cyrillic = {
            'a': 'а', 'b': 'б', 'v': 'в', 'd': 'д', 'e': 'е', 'f': 'ф',
            'g': 'г', 'h': 'ҳ', 'i': 'и', 'j': 'ж', 'k': 'к', 'l': 'л',
            'm': 'м', 'n': 'н', 'o': 'о', 'p': 'п', 'q': 'қ', 'r': 'р',
            's': 'с', 't': 'т', 'u': 'у', 'v': 'в', 'x': 'х', 'y': 'й',
            'z': 'з', "o'": 'ў', "g'": 'ғ', 'sh': 'ш', 'ch': 'ч', 'ng': 'нг',
            'A': 'А', 'B': 'Б', 'V': 'В', 'D': 'Д', 'E': 'Е', 'F': 'Ф',
            'G': 'Г', 'H': 'Ҳ', 'I': 'И', 'J': 'Ж', 'K': 'К', 'L': 'Л',
            'M': 'М', 'N': 'Н', 'O': 'О', 'P': 'П', 'Q': 'Қ', 'R': 'Р',
            'S': 'С', 'T': 'Т', 'U': 'У', 'V': 'В', 'X': 'Х', 'Y': 'Й',
            'Z': 'З', "O'": 'Ў', "G'": 'Ғ', 'Sh': 'Ш', 'Ch': 'Ч', 'Ng': 'Нг',
            'SH': 'Ш', 'CH': 'Ч', 'NG': 'НГ'
        }
        
        # Cyrillic to Latin mapping for Uzbek
        self.cyrillic_to_latin = {v: k for k, v in self.latin_to_cyrillic.items()}
    
    def detect_script(self, text: str) -> str:
        """Detect if text is Latin or Cyrillic."""
        cyrillic_chars = sum(1 for c in text if '\u0400' <= c <= '\u04FF')
        latin_chars = sum(1 for c in text if 'a' <= c.lower() <= 'z')
        
        if cyrillic_chars > latin_chars:
            return 'Cyrillic'
        return 'Latin'
    
    def transliterate_latin_to_cyrillic(self, text: str) -> str:
        """Convert Latin Uzbek to Cyrillic."""
        result = text
        
        # Replace digraphs first (order matters)
        for lat, cyr in [('sh', 'ш'), ('ch', 'ч'), ('ng', 'нг'), 
                         ('Sh', 'Ш'), ('Ch', 'Ч'), ('Ng', 'Нг'),
                         ('SH', 'Ш'), ('CH', 'Ч'), ('NG', 'НГ'),
                         ("o'", 'ў'), ("g'", 'ғ'), ("O'", 'Ў'), ("G'", 'Ғ')]:
            result = result.replace(lat, cyr)
        
        # Replace single characters
        for lat, cyr in self.latin_to_cyrillic.items():
            if len(lat) == 1:
                result = result.replace(lat, cyr)
        
        return result
    
    def transliterate_cyrillic_to_latin(self, text: str) -> str:
        """Convert Cyrillic Uzbek/Russian to Latin."""
        result = text
        
        # Replace Cyrillic characters
        for cyr, lat in self.cyrillic_to_latin.items():
            result = result.replace(cyr, lat)
        
        return result
    
    def chunk_all_documents(self, output_dir: str = 'chunked_output'):
        """Process all active documents."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("="*80)
        print("STRICT PARAGRAPH-BASED CHUNKING WITH TRANSLITERATION")
        print("="*80)
        print("Rules:")
        print("  - New line + capital letter = new chunk")
        print("  - Line with 2+ leading spaces = new chunk")
        print("  - Section headers excluded from chunks")
        print("  - Stores both Latin and Cyrillic versions")
        print("="*80)
        
        active_docs = self.metadata_df[self.metadata_df['Status'] == 'Active'].copy()
        print(f"\n📄 Found {len(active_docs)} active documents")
        
        proceed = input("\n⚠️  Proceed? (yes/no): ").strip().lower()
        if proceed != 'yes':
            return []
        
        print("\n" + "="*80)
        print("PROCESSING")
        print("="*80)
        
        for idx, row in active_docs.iterrows():
            print(f"\n[{idx+1}/{len(active_docs)}] {row['Filename']}")
            try:
                chunks = self.chunk_single_document(row)
                self.chunks_output.extend(chunks)
                print(f"   ✓ Generated {len(chunks)} chunks")
            except Exception as e:
                print(f"   ✗ Error: {str(e)}")
        
        self.save_chunks(output_path)
        self.generate_statistics(output_path)
        return self.chunks_output
    
    def chunk_single_document(self, metadata_row: pd.Series) -> List[Dict]:
        """Process single PDF."""
        pdf_path = Path(metadata_row['Full_Path'])
        
        print(f"   [1/3] Extracting text...")
        full_text = self.extract_text_from_pdf(pdf_path)
        
        if not full_text or len(full_text.strip()) < 100:
            raise ValueError("Empty PDF")
        
        print(f"   [2/3] Splitting into chunks...")
        chunks_text = self.split_into_chunks(full_text)
        
        print(f"   [3/3] Creating chunk objects...")
        chunks = []
        
        for i, chunk_text in enumerate(chunks_text):
            section_num = self.extract_section_number(chunk_text)
            chunk = self.create_chunk(chunk_text, i, metadata_row, len(chunks_text), section_num)
            chunks.append(chunk)
        
        return chunks
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text, clean page numbers."""
        try:
            reader = PdfReader(str(pdf_path))
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    # Remove standalone page numbers
                    page_text = re.sub(r'^\s*\d+\s*$', '', page_text, flags=re.MULTILINE)
                    text += page_text + "\n"
            return text
        except Exception as e:
            raise ValueError(f"PDF extraction failed: {str(e)}")
    
    def split_into_chunks(self, text: str, min_words: int = 10) -> List[str]:
        """
        Strict chunking rules:
        1. Split on lines that start with capital letter (new sentence on new line)
        2. Split on lines that start with 2+ spaces
        3. Remove section headers (lines matching section number pattern)
        """
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        
        # Pattern for section headers (e.g., "1.", "1.2.", "1.2.3.")
        section_header_pattern = r'^\s*\d+\.(?:\d+\.)*\s+[А-ЯЁA-Z]'
        
        for line in lines:
            line_stripped = line.strip()
            
            # Skip empty lines
            if not line_stripped:
                continue
            
            # Skip section headers entirely
            if re.match(section_header_pattern, line_stripped):
                # Save current chunk before skipping header
                if current_chunk:
                    chunk_text = ' '.join(current_chunk).strip()
                    if len(chunk_text.split()) >= min_words:
                        chunks.append(chunk_text)
                    current_chunk = []
                continue
            
            # Check if line starts with 2+ spaces
            starts_with_spaces = len(line) - len(line.lstrip()) >= 2
            
            # Check if line starts with capital letter (Cyrillic or Latin)
            starts_with_capital = bool(re.match(r'^[А-ЯЁA-Z]', line_stripped))
            
            # New chunk conditions
            should_split = starts_with_spaces or (starts_with_capital and current_chunk)
            
            if should_split and current_chunk:
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
        
        return chunks
    
    def extract_section_number(self, text: str) -> str:
        """Extract section number if present at start of chunk."""
        # Look for section number at the very beginning
        match = re.match(r'^(\d+\.(?:\d+\.)*)', text)
        if match:
            return match.group(1)
        return 'unknown'
    
    def create_chunk(self, paragraph_text: str, chunk_index: int,
                    metadata_row: pd.Series, total_chunks: int,
                    section_number: str) -> Dict:
        """Create enriched chunk with transliteration."""
        hotspots = self.detect_compliance_hotspots(paragraph_text)
        
        # Detect script and create both versions
        detected_script = self.detect_script(paragraph_text)
        
        if detected_script == 'Latin':
            chunk_text_latin = paragraph_text
            chunk_text_cyrillic = self.transliterate_latin_to_cyrillic(paragraph_text)
        else:  # Cyrillic
            chunk_text_cyrillic = paragraph_text
            chunk_text_latin = self.transliterate_cyrillic_to_latin(paragraph_text)
        
        return {
            'chunk_id': f"{metadata_row['Filename'].replace('.pdf', '')}__chunk_{chunk_index}",
            'document_filename': metadata_row['Filename'],
            'document_date': metadata_row['Version_Date'],
            'document_status': metadata_row['Status'],
            'document_language': metadata_row['Language'],
            'document_script': metadata_row['Script'],
            'document_folder': metadata_row['Folder'],
            'section_number': section_number,
            'chunk_index': chunk_index,
            'total_chunks': total_chunks,
            'chunk_position': f"{chunk_index + 1}/{total_chunks}",
            'chunk_text_latin': chunk_text_latin,
            'chunk_text_cyrillic': chunk_text_cyrillic,
            'detected_script': detected_script,
            'chunk_length': len(paragraph_text),
            'chunk_words': len(paragraph_text.split()),
            'contains_interest_rate': hotspots['interest_rate'],
            'contains_collateral': hotspots['collateral'],
            'contains_mandatory_disclosure': hotspots['mandatory_disclosure'],
            'contains_preselection': hotspots['pre_selected'],
            'contains_language_req': hotspots['language_requirement'],
            'compliance_flags': [k for k, v in hotspots.items() if v],
            'ready_for_embedding': True
        }
    
    def detect_compliance_hotspots(self, text: str) -> Dict[str, bool]:
        """Detect compliance keywords."""
        text_lower = text.lower()
        hotspots = {}
        
        for category, keywords in self.compliance_keywords.items():
            hotspots[category] = any(kw.lower() in text_lower for kw in keywords)
        
        return hotspots
    
    def save_chunks(self, output_path: Path):
        """Save outputs."""
        json_file = output_path / 'bank_chunks_metadata.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.chunks_output, f, ensure_ascii=False, indent=2)
        print(f"\n✓ Saved: {json_file}")
        
        chunks_df = pd.DataFrame(self.chunks_output)
        csv_file = output_path / 'bank_chunks_index.csv'
        
        key_cols = [
            'chunk_id', 'document_filename', 'document_date', 'document_language',
            'section_number', 'detected_script', 'chunk_words', 'compliance_flags'
        ]
        chunks_df[key_cols].to_csv(csv_file, index=False, encoding='utf-8')
        print(f"✓ Saved: {csv_file}")
    
    def generate_statistics(self, output_path: Path):
        """Generate statistics."""
        stats = {
            'total_chunks': len(self.chunks_output),
            'total_documents': len(self.metadata_df[self.metadata_df['Status'] == 'Active']),
            'avg_chunks_per_doc': len(self.chunks_output) / max(1, len(self.metadata_df[self.metadata_df['Status'] == 'Active'])),
            'avg_words_per_chunk': sum(c['chunk_words'] for c in self.chunks_output) / max(1, len(self.chunks_output)),
            'language_distribution': {},
            'script_distribution': {},
            'compliance_distribution': {}
        }
        
        for chunk in self.chunks_output:
            lang = chunk['document_language']
            stats['language_distribution'][lang] = stats['language_distribution'].get(lang, 0) + 1
            
            script = chunk['detected_script']
            stats['script_distribution'][script] = stats['script_distribution'].get(script, 0) + 1
        
        for category in self.compliance_keywords.keys():
            count = sum(1 for c in self.chunks_output if c.get(f'contains_{category}', False))
            stats['compliance_distribution'][category] = count
        
        stats_file = output_path / 'chunking_statistics.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print("\n" + "="*80)
        print("STATISTICS")
        print("="*80)
        print(f"Total chunks: {stats['total_chunks']}")
        print(f"Documents: {stats['total_documents']}")
        print(f"Avg chunks/doc: {stats['avg_chunks_per_doc']:.1f}")
        print(f"Avg words/chunk: {stats['avg_words_per_chunk']:.1f}")
        
        print(f"\n📊 By language:")
        for lang, count in stats['language_distribution'].items():
            print(f"  {lang}: {count}")
        
        print(f"\n📝 By script:")
        for script, count in stats['script_distribution'].items():
            print(f"  {script}: {count}")
        
        print(f"\n🚨 Compliance hotspots:")
        for cat, count in stats['compliance_distribution'].items():
            print(f"  {cat}: {count}")
        
        return stats


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    metadata_path = script_dir / 'bank_policies' / 'anorbank' / 'metadata' / 'anorbank_index.csv'
    
    if not metadata_path.exists():
        print(f"ERROR: Metadata not found at {metadata_path}")
        exit(1)
    
    print(f"✓ Found metadata: {metadata_path}")
    
    chunker = ParagraphDocumentChunker(str(metadata_path))
    output_dir = script_dir / 'chunked_output'
    
    chunks = chunker.chunk_all_documents(output_dir=str(output_dir))
    
    print("\n" + "="*80)
    print("✓ COMPLETE")
    print("="*80)
