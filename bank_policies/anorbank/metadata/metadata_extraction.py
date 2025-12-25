import os
import re
import pandas as pd
from pathlib import Path  # More Pythonic, works great on Linux

def extract_metadata():
    """
    Extracts metadata from PDF files in the anorbank directory structure.
    Optimized for Linux filesystems (ext4, btrfs, xfs).
    """
    # Using pathlib (more Pythonic and cleaner on Linux)
    script_path = Path(__file__).resolve()
    base_path = script_path.parent.parent
    
    print(f"--- Scanning Directory: {base_path} ---\n")
    
    data = []
    
    # Enhanced regex patterns for various date formats
    date_patterns = [
        r'(\d{2}[._]\d{2}[._]\d{4})',  # DD.MM.YYYY or DD_MM_YYYY
        r'(\d{2}\.\d{2}\.\d{4})',       # DD.MM.YYYY with dots
        r'(\d{2}_\d{2}_\d{4})',         # DD_MM_YYYY with underscores
        r'от[_\s]+(\d{2})[_\s]+(\d{2})[_\s]+(\d{4})',  # от_DD_MM_YYYY (Cyrillic)
        r'dan[_\s]+(\d{2})[._](\d{2})[._](\d{4})',     # dan DD.MM.YYYY
    ]
    
    if not base_path.exists():
        print(f"Critical Error: Path {base_path} not found!")
        return pd.DataFrame()
    
    # Walk through directory structure
    for pdf_file in base_path.rglob('*.pdf'):  # Recursive glob - very Linux-friendly
        # Skip metadata folder and hidden folders
        if 'metadata' in pdf_file.parts or any(part.startswith('.') for part in pdf_file.parts):
            continue
        
        filename = pdf_file.name
        
        # Extract date from filename
        extracted_date = extract_date_from_filename(filename, date_patterns)
        
        # Detect Status (Uzbek: Toxtatildi = Stopped)
        status = "Stopped (Inactive)" if "[Toxtatildi]" in filename or "Toxtatildi" in filename else "Active"
        
        # Detect Script (Cyrillic vs Latin)
        script_type = detect_script(filename)
        
        # Get relative folder path
        relative_folder = pdf_file.parent.relative_to(base_path)
        
        data.append({
            "Filename": filename,
            "Version_Date": extracted_date,
            "Status": status,
            "Script": script_type,
            "Folder": str(relative_folder),
            "Full_Path": str(pdf_file)
        })
    
    # Safety check
    cols = ['Version_Date', 'Status', 'Script', 'Filename', 'Folder']
    if not data:
        print("Warning: No PDF files found! Check if folders are empty.")
        return pd.DataFrame(columns=cols)
    
    return pd.DataFrame(data)


def extract_date_from_filename(filename, patterns):
    """
    Extracts date from filename using multiple regex patterns.
    Returns normalized date string in DD.MM.YYYY format.
    """
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            if len(match.groups()) == 1:
                raw_date = match.group(1)
                clean_date = raw_date.replace('_', '.')
                return clean_date
            elif len(match.groups()) == 3:
                day, month, year = match.groups()
                return f"{day}.{month}.{year}"
    
    return "Unknown"


def detect_script(filename):
    """
    Detects if the filename uses Cyrillic or Latin script.
    """
    has_cyrillic = bool(re.search(r'[\u0400-\u04FF]', filename))
    has_latin = bool(re.search(r'[a-zA-Z]', filename))
    
    if has_cyrillic and has_latin:
        return "Mixed (Cyrillic + Latin)"
    elif has_cyrillic:
        return "Uzbek Cyrillic"
    elif has_latin:
        return "Uzbek Latin"
    else:
        return "Unknown"


def parse_date_safely(date_str):
    """
    Safely parse date string to datetime object.
    """
    if date_str == "Unknown":
        return None
    
    try:
        return pd.to_datetime(date_str, format='%d.%m.%Y', errors='coerce')
    except:
        return None


# --- EXECUTION ---
if __name__ == "__main__":
    metadata_df = extract_metadata()
    
    if not metadata_df.empty:
        # Smart sorting
        metadata_df['sort_date'] = metadata_df['Version_Date'].apply(parse_date_safely)
        
        result = metadata_df.sort_values(
            by=['sort_date', 'Folder'], 
            ascending=[False, True]
        )
        
        # Display results
        display_cols = ['Version_Date', 'Status', 'Script', 'Filename', 'Folder']
        print("\n" + "="*100)
        print("METADATA EXTRACTION RESULTS")
        print("="*100 + "\n")
        print(result[display_cols].to_string(index=False))
        
        # Summary statistics
        print("\n" + "="*100)
        print("SUMMARY STATISTICS")
        print("="*100)
        print(f"Total PDF files found: {len(result)}")
        print(f"Active documents: {len(result[result['Status'] == 'Active'])}")
        print(f"Stopped documents: {len(result[result['Status'] == 'Stopped (Inactive)'])}")
        print(f"\nDocuments by script:")
        print(result['Script'].value_counts().to_string())
        print(f"\nDocuments by folder:")
        print(result['Folder'].value_counts().to_string())
        
        # Save to CSV (Linux-friendly path)
        output_path = Path(__file__).parent / 'anorbank_index.csv'
        result[display_cols].to_csv(output_path, index=False, encoding='utf-8')
        print(f"\n✓ Results saved to: {output_path}")
        
    else:
        print("No data to display. Please check file paths.")
