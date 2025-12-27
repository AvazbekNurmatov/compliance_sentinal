import os
import json
import pandas as pd
from pathlib import Path
from PyPDF2 import PdfReader
from typing import List, Dict
import re
from datetime import datetime

class RegulationChunker:
    """
    Strict regulation chunking logic:
    1. If a line ends with . or ; -> Next line starts a new chunk (regardless of casing).
    2. If a line starts with 2+ spaces -> New chunk.
    3. Section headers (e.g., 1.2.3.) -> Excluded from text, trigger new chunk.
    4. Stores both Latin and Cyrillic versions for searchability.
    """
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.regulations_dir = base_dir / 'regulations'
        self.output_dir = base_dir / 'chunked_output' / 'chunked_regulations'
        
        # Mapping for Uzbek Transliteration
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
        self.cyrillic_to_latin = {v: k for k, v in self.latin_to_cyrillic.items()}

    def detect_script(self, text: str) -> str:
        cyrillic_chars = sum(1 for c in text if '\u0400' <= c <= '\u04FF')
        latin_chars = sum(1 for c in text if 'a' <= c.lower() <= 'z')
        return 'Cyrillic' if cyrillic_chars > latin_chars else 'Latin'

    def transliterate(self, text: str, to_script: str) -> str:
        result = text
        # Order matters for digraphs (sh, ch, o', g')
        digraphs = [
            ('sh', 'ш'), ('ch', 'ч'), ('ng', 'нг'), ("o'", 'ў'), ("g'", 'ғ'),
            ('Sh', 'Ш'), ('Ch', 'Ч'), ('Ng', 'Нг'), ("O'", 'Ў'), ("G'", 'Ғ'),
            ('SH', 'Ш'), ('CH', 'Ч'), ('NG', 'НГ')
        ]
        
        if to_script == 'Cyrillic':
            for lat, cyr in digraphs:
                result = result.replace(lat, cyr)
            for lat, cyr in self.latin_to_cyrillic.items():
                if len(lat) == 1: result = result.replace(lat, cyr)
        else:
            for lat, cyr in digraphs:
                result = result.replace(cyr, lat)
            for cyr, lat in self.cyrillic_to_latin.items():
                result = result.replace(cyr, lat)
        return result

    def split_into_chunks(self, text: str, min_words: int = 5) -> List[str]:
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        
        # State: should the next valid line be a new chunk?
        force_new_chunk = False
        section_pattern = r'^\s*\d+\.(?:\d+\.)*\s+[А-ЯЁA-Z]'

        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue

            # Rule 1: Remove/Split on Section Headers
            if re.match(section_pattern, line_stripped):
                self._flush_chunk(current_chunk, chunks, min_words)
                current_chunk = []
                force_new_chunk = False
                continue

            # Rule 2: Check for splits
            # - Starts with 2+ spaces
            # - OR The previous line ended with . or ;
            starts_with_spaces = len(line) - len(line.lstrip()) >= 2
            
            if (force_new_chunk or starts_with_spaces) and current_chunk:
                self._flush_chunk(current_chunk, chunks, min_words)
                current_chunk = []

            # Add current line to active chunk
            current_chunk.append(line_stripped)

            # Update State: If this line ends with . or ;, force next line to split
            force_new_chunk = line_stripped.endswith(('.', ';'))

        # Final flush
        self._flush_chunk(current_chunk, chunks, min_words)
        return chunks

    def _flush_chunk(self, current_chunk, chunks, min_words):
        if current_chunk:
            text = ' '.join(current_chunk).strip()
            if len(text.split()) >= min_words:
                chunks.append(text)

    def process_regulations(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        all_chunks = []
        
        pdf_files = list(self.regulations_dir.glob("*.pdf"))
        print(f"🚀 Found {len(pdf_files)} regulations to process.")

        for pdf_path in pdf_files:
            print(f"📄 Processing: {pdf_path.name}")
            try:
                reader = PdfReader(pdf_path)
                full_text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        # Clean standalone page numbers
                        page_text = re.sub(r'^\s*\d+\s*$', '', page_text, flags=re.MULTILINE)
                        full_text += page_text + "\n"
                
                raw_chunks = self.split_into_chunks(full_text)
                
                for i, text in enumerate(raw_chunks):
                    script = self.detect_script(text)
                    all_chunks.append({
                        'chunk_id': f"reg_{pdf_path.stem}_{i}",
                        'document_filename': pdf_path.name,
                        'chunk_index': i,
                        'chunk_text_latin': text if script == 'Latin' else self.transliterate(text, 'Latin'),
                        'chunk_text_cyrillic': text if script == 'Cyrillic' else self.transliterate(text, 'Cyrillic'),
                        'detected_script': script,
                        'metadata': {
                            'source_folder': 'regulations',
                            'word_count': len(text.split()),
                            'processed_date': datetime.now().strftime("%Y-%m-%d")
                        }
                    })
                print(f"   ✅ Generated {len(raw_chunks)} chunks")
            except Exception as e:
                print(f"   ❌ Error: {e}")

        # Save to regulation_embeddings folder
        json_out = self.output_dir / 'regulation_chunks_metadata.json'
        with open(json_out, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)
            
        # Also save a CSV index for quick viewing
        csv_out = self.output_dir / 'regulation_chunks_index.csv'
        pd.DataFrame(all_chunks).drop(columns=['chunk_text_latin', 'chunk_text_cyrillic']).to_csv(csv_out, index=False)
        
        print(f"\n✨ DONE. Data stored in: {self.output_dir}")

if __name__ == "__main__":
    # Get the directory where the script is located
    root_dir = Path(__file__).resolve().parent
    chunker = RegulationChunker(root_dir)
    chunker.process_regulations()
