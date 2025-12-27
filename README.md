# 🏦 Bank Compliance Sentinel

An AI-powered compliance checking system for bank contracts and documents. This tool analyzes loan contracts against Central Bank of Uzbekistan (CBU) regulations and Anor Bank policies to ensure compliance and provide actionable corrections.

## 🎯 Features

- **PDF Processing**: Extract and chunk text from bank contracts
- **Semantic Search**: Compare documents against CBU regulations and bank policies using embeddings
- **Compliance Checking**: Identify violations with similarity scoring
- **Correction Generation**: Provide actionable fixes for non-compliant sections
- **Web Interface**: User-friendly Streamlit dashboard
- **Multi-language Support**: Handles Uzbek (Latin/Cyrillic) and Russian documents

## 🛠 Tech Stack

- **Backend**: Python 3.8+
- **Vector Database**: ChromaDB for semantic search
- **Embeddings**: OpenAI text-embedding-3-small
- **PDF Processing**: PyMuPDF (fitz), pdfplumber
- **Web UI**: Streamlit
- **AI/ML**: LangChain, OpenAI API

## 📁 Project Structure

```
compliance_sentinal/
├── 📄 Core Files
│   ├── app.py                    # Streamlit web interface
│   ├── compliance_checker.py     # Main compliance analysis engine
│   ├── pdf_processor.py          # PDF text extraction & chunking
│   ├── correction_generator.py   # Generate actionable corrections
│   └── requirements.txt          # Python dependencies
│
├── 📊 Data Processing
│   ├── chroma_ingestion.py       # ChromaDB data ingestion
│   ├── chunk_regulations.py      # Process CBU regulations
│   ├── chunking_script.py       # Generic document chunking
│   ├── embed_regulations.py      # Generate embeddings for regulations
│   └── embedding_script.py       # Generic embedding generation
│
├── 📋 Reference Documents
│   ├── bank_policies/            # Anor Bank policy documents
│   │   └── anorbank/
│   │       ├── contracts/        # Universal contracts
│   │       ├── metadata/         # CSV indexes and extraction tools
│   │       └── promotions/       # Promotional materials
│   └── regulations/              # CBU regulations and laws
│
├── 💾 Vector Storage
│   └── index/
│       └── chroma_db/           # ChromaDB persistent storage
│
├── 📄 Sample Files
│   ├── sample_bank_paper.pdf     # Sample contract for testing
│   └── sample_bank_paper.docx    # Sample document
│
└── 🔧 Utilities
    ├── check_chromadb.py         # Verify ChromaDB setup
    └── .gitignore               # Git ignore rules
```

## 🚀 Setup Instructions

### 1. Prerequisites
- Python 3.8 or higher
- OpenAI API key
- Git

### 2. Clone and Install
```bash
git clone <repository-url>
cd compliance_sentinal
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Data Setup
The system comes pre-loaded with:
- CBU regulations in `regulations/`
- Anor Bank policies in `bank_policies/`
- Pre-computed embeddings in `embeddings/`

If you need to re-index the data:
```bash
# Re-ingest documents into ChromaDB
python chroma_ingestion.py

# Process new regulations
python chunk_regulations.py
python embed_regulations.py
```

## 🏃 How to Run

### Web Interface (Recommended)
```bash
streamlit run app.py
```
Navigate to `http://localhost:8501` to access the web dashboard.

### Command Line Interface
```bash
# Process a PDF and check compliance
python pdf_processor.py        # Process uploaded PDF
python compliance_checker.py   # Run compliance analysis
python correction_generator.py # Generate correction report
```

## 📖 Example Usage

### Via Web Interface
1. Launch the web app: `streamlit run app.py`
2. Upload a PDF contract using the file uploader
3. Click "🔍 Check Compliance" to analyze
4. Review the compliance report with scores and violations
5. Click "🔧 Show Corrections" for actionable fixes
6. Download reports as JSON files

### Via Command Line
```bash
# Process a sample contract
python pdf_processor.py

# Check compliance against regulations and policies
python compliance_checker.py

# Generate detailed correction report
python correction_generator.py
```

## 📊 Compliance Reports

The system generates comprehensive reports including:

- **Overall Compliance Score** (0-100%)
- **Status Classification** (PASS/WARNING/FAIL)
- **Regulation Violations** with specific CBU references
- **Policy Violations** with bank policy sources
- **Actionable Corrections** categorized by severity:
  - 🔴 **Critical**: Must fix immediately
  - 🟠 **High Priority**: Important to address
  - 🟡 **Medium Priority**: Should be corrected

## 🔍 What It Checks

### CBU Regulation Compliance
- Capital requirements
- Loan-to-value ratios
- Interest rate limits
- Collateral requirements
- Documentation standards

### Bank Policy Compliance
- Document submission timelines
- Monitoring frequencies
- Client notification periods
- Mandatory clauses
- Contract formatting standards

## 🛠 Development

### Adding New Documents
1. Place PDFs in appropriate folders (`bank_policies/` or `regulations/`)
2. Update CSV indexes in `metadata/`
3. Re-run chunking and embedding scripts
4. Re-ingest into ChromaDB

### Customizing Thresholds
Edit similarity thresholds in `compliance_checker.py`:
```python
self.STRONG_MATCH = 0.3      # Distance < 0.3 = good match
self.WEAK_MATCH = 0.5        # Distance 0.3-0.5 = weak match
self.NO_MATCH = 0.5          # Distance > 0.5 = no match/violation
```

### Extending Correction Patterns
Add new number patterns in `correction_generator.py`:
```python
self.number_patterns = {
    'days': r'(\d+)\s*(?:kalendar|календар|ish|иш)?\s*(?:kun|кун|day)',
    'percent': r'(\d+(?:\.\d+)?)\s*(?:%|процент|foiz|фоиз)',
    # Add your patterns here
}
```

## 🐛 Troubleshooting

### Common Issues
1. **ChromaDB Connection Error**: Ensure `index/chroma_db/` exists and is accessible
2. **OpenAI API Error**: Verify API key in `.env` file
3. **PDF Processing Error**: Check if PDF is text-based (not scanned images)
4. **No Matches Found**: Re-run `chroma_ingestion.py` to populate the database

### Verification Commands
```bash
# Check ChromaDB status
python check_chromadb.py

# Verify embeddings
python -c "import chromadb; client = chromadb.PersistentClient('./index/chroma_db'); print('Collections:', client.list_collections())"
```

## 📄 License

This project is proprietary and confidential. Use only with authorized access.

## 🤝 Support

For technical support or questions:
- Check the troubleshooting section above
- Review the code comments for detailed explanations
- Contact the development team

---

**⚡ Quick Start**: `streamlit run app.py` → Upload PDF → Get Compliance Report!