"""
Compliance Checker
Compares uploaded PDF chunks against regulations and bank policies
"""

import chromadb
from typing import List, Dict, Tuple
import re
from datetime import datetime

class ComplianceChecker:
    """Check uploaded documents for compliance issues"""
    
    def __init__(self, chroma_path: str = "./index/chroma_db"):
        """Initialize with ChromaDB collections"""
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        
        # Get existing collections
        try:
            self.regulations_collection = self.chroma_client.get_collection("regulations")
            print(f"✅ Loaded regulations collection: {self.regulations_collection.count()} items")
        except Exception as e:
            print(f"⚠️  Warning: regulations collection not found")
            self.regulations_collection = None
        
        try:
            self.policies_collection = self.chroma_client.get_collection("bank_policies")
            print(f"✅ Loaded bank_policies collection: {self.policies_collection.count()} items")
        except Exception as e:
            print(f"⚠️  Warning: bank_policies collection not found")
            self.policies_collection = None
        
        try:
            self.uploaded_collection = self.chroma_client.get_collection("uploaded_documents")
            print(f"✅ Loaded uploaded_documents collection: {self.uploaded_collection.count()} items")
        except Exception as e:
            print(f"❌ Error: uploaded_documents collection not found")
            self.uploaded_collection = None
        
        # Similarity thresholds
        self.STRONG_MATCH = 0.3      # Distance < 0.3 = good match
        self.WEAK_MATCH = 0.5        # Distance 0.3-0.5 = weak match
        self.NO_MATCH = 0.5          # Distance > 0.5 = no match/violation
        
        # Number patterns for detecting violations
        self.number_patterns = {
            'days': r'(\d+)\s*(?:календар|kalendar|ish|иш)?\s*(?:kun|кун|day)',
            'percent': r'(\d+(?:\.\d+)?)\s*(?:%|процент|foiz|фоиз)',
            'months': r'(\d+)\s*(?:oy|ой|месяц|month)',
            'years': r'(\d+)\s*(?:yil|йил|год|year)'
        }
    
    def get_uploaded_chunks(self, doc_id: str = None) -> List[Dict]:
        """
        Get chunks from uploaded document
        
        Args:
            doc_id: Specific document ID, or None for latest
            
        Returns:
            List of chunk dictionaries
        """
        if not self.uploaded_collection:
            raise Exception("No uploaded documents found")
        
        # Get all uploaded chunks
        results = self.uploaded_collection.get(
            include=["embeddings", "documents", "metadatas"]
        )
        
        chunks = []
        for i in range(len(results['ids'])):
            chunk = {
                'id': results['ids'][i],
                'text': results['documents'][i],
                'embedding': results['embeddings'][i],
                'metadata': results['metadatas'][i]
            }
            chunks.append(chunk)
        
        # Sort by chunk_index
        chunks.sort(key=lambda x: x['metadata'].get('chunk_index', 0))
        
        print(f"📄 Retrieved {len(chunks)} chunks from uploaded document")
        return chunks
    
    def check_against_regulations(self, uploaded_chunks: List[Dict]) -> Dict:
        """
        Compare uploaded chunks against CBU regulations
        
        Args:
            uploaded_chunks: List of uploaded document chunks
            
        Returns:
            Dictionary with compliance results
        """
        if not self.regulations_collection:
            return {"error": "Regulations collection not available"}
        
        print(f"\n{'='*60}")
        print("🔍 CHECKING AGAINST CBU REGULATIONS")
        print(f"{'='*60}\n")
        
        results = {
            "total_chunks_checked": len(uploaded_chunks),
            "matches": [],
            "weak_matches": [],
            "violations": [],
            "summary": {}
        }
        
        for i, chunk in enumerate(uploaded_chunks):
            print(f"  Checking chunk {i+1}/{len(uploaded_chunks)}...", end="\r")
            
            # Query regulations for similar content
            reg_results = self.regulations_collection.query(
                query_embeddings=[chunk['embedding']],
                n_results=3,
                include=["documents", "metadatas", "distances"]
            )
            
            # Analyze matches
            for j in range(len(reg_results['ids'][0])):
                distance = reg_results['distances'][0][j]
                matched_text = reg_results['documents'][0][j]
                matched_metadata = reg_results['metadatas'][0][j]
                
                # Skip if matched_text is None
                if matched_text is None:
                    continue
                
                match_info = {
                    'uploaded_chunk_id': chunk['id'],
                    'uploaded_text': chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text'],
                    'matched_regulation': matched_text[:200] + "..." if len(matched_text) > 200 else matched_text,
                    'regulation_source': matched_metadata.get('document_filename', 'Unknown'),
                    'similarity_score': round(1 - distance, 3),  # Convert distance to similarity
                    'distance': round(distance, 3),
                    'potential_issue': None
                }
                
                # Check for numerical discrepancies
                issue = self._check_numerical_discrepancy(chunk['text'], matched_text)
                if issue:
                    match_info['potential_issue'] = issue
                
                # Categorize match
                if distance < self.STRONG_MATCH:
                    results['matches'].append(match_info)
                elif distance < self.WEAK_MATCH:
                    results['weak_matches'].append(match_info)
                else:
                    if match_info['potential_issue']:
                        results['violations'].append(match_info)
        
        print(f"  Checking chunk {len(uploaded_chunks)}/{len(uploaded_chunks)}... Done!")
        
        # Generate summary
        results['summary'] = {
            'total_strong_matches': len(results['matches']),
            'total_weak_matches': len(results['weak_matches']),
            'total_violations': len(results['violations']),
            'compliance_score': self._calculate_compliance_score(results)
        }
        
        return results
    
    def check_against_policies(self, uploaded_chunks: List[Dict]) -> Dict:
        """
        Compare uploaded chunks against bank policies
        
        Args:
            uploaded_chunks: List of uploaded document chunks
            
        Returns:
            Dictionary with compliance results
        """
        if not self.policies_collection:
            return {"error": "Bank policies collection not available"}
        
        print(f"\n{'='*60}")
        print("🔍 CHECKING AGAINST BANK POLICIES")
        print(f"{'='*60}\n")
        
        results = {
            "total_chunks_checked": len(uploaded_chunks),
            "matches": [],
            "weak_matches": [],
            "violations": [],
            "missing_clauses": [],
            "summary": {}
        }
        
        for i, chunk in enumerate(uploaded_chunks):
            print(f"  Checking chunk {i+1}/{len(uploaded_chunks)}...", end="\r")
            
            # Query policies for similar content
            policy_results = self.policies_collection.query(
                query_embeddings=[chunk['embedding']],
                n_results=3,
                include=["documents", "metadatas", "distances"]
            )
            
            # Analyze matches
            for j in range(len(policy_results['ids'][0])):
                distance = policy_results['distances'][0][j]
                matched_text = policy_results['documents'][0][j]
                matched_metadata = policy_results['metadatas'][0][j]
                
                # Skip if matched_text is None
                if matched_text is None:
                    continue
                
                match_info = {
                    'uploaded_chunk_id': chunk['id'],
                    'uploaded_text': chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text'],
                    'matched_policy': matched_text[:200] + "..." if len(matched_text) > 200 else matched_text,
                    'policy_source': matched_metadata.get('document_filename', 'Unknown'),
                    'similarity_score': round(1 - distance, 3),
                    'distance': round(distance, 3),
                    'potential_issue': None
                }
                
                # Check for numerical discrepancies
                issue = self._check_numerical_discrepancy(chunk['text'], matched_text)
                if issue:
                    match_info['potential_issue'] = issue
                
                # Categorize match
                if distance < self.STRONG_MATCH:
                    results['matches'].append(match_info)
                elif distance < self.WEAK_MATCH:
                    results['weak_matches'].append(match_info)
                else:
                    if match_info['potential_issue']:
                        results['violations'].append(match_info)
        
        print(f"  Checking chunk {len(uploaded_chunks)}/{len(uploaded_chunks)}... Done!")
        
        # Generate summary
        results['summary'] = {
            'total_strong_matches': len(results['matches']),
            'total_weak_matches': len(results['weak_matches']),
            'total_violations': len(results['violations']),
            'compliance_score': self._calculate_compliance_score(results)
        }
        
        return results
    
    def _check_numerical_discrepancy(self, text1: str, text2: str) -> str:
        """
        Check if two texts have different numbers for same concepts
        
        Args:
            text1: First text (uploaded)
            text2: Second text (reference)
            
        Returns:
            Issue description or None
        """
        issues = []
        
        for concept, pattern in self.number_patterns.items():
            # Extract numbers from both texts
            nums1 = re.findall(pattern, text1.lower())
            nums2 = re.findall(pattern, text2.lower())
            
            if nums1 and nums2:
                # Convert to floats for comparison
                try:
                    vals1 = [float(n) for n in nums1]
                    vals2 = [float(n) for n in nums2]
                    
                    # Check if any values differ
                    for v1 in vals1:
                        for v2 in vals2:
                            if v1 != v2:
                                issues.append(f"{concept.upper()}: Uploaded says {v1}, reference says {v2}")
                except:
                    pass
        
        return " | ".join(issues) if issues else None
    
    def _calculate_compliance_score(self, results: Dict) -> float:
        """
        Calculate overall compliance score (0-100)
        
        Args:
            results: Compliance check results
            
        Returns:
            Compliance score percentage
        """
        total = len(results['matches']) + len(results['weak_matches']) + len(results['violations'])
        
        if total == 0:
            return 0.0
        
        # Strong matches = 1.0, weak matches = 0.5, violations = 0.0
        score = (len(results['matches']) * 1.0 + len(results['weak_matches']) * 0.5) / total
        return round(score * 100, 2)
    
    def generate_report(self, regulation_results: Dict, policy_results: Dict) -> Dict:
        """
        Generate comprehensive compliance report
        
        Args:
            regulation_results: Results from regulation check
            policy_results: Results from policy check
            
        Returns:
            Complete compliance report
        """
        print(f"\n{'='*60}")
        print("📋 GENERATING COMPLIANCE REPORT")
        print(f"{'='*60}\n")
        
        # Determine overall status
        reg_score = regulation_results['summary'].get('compliance_score', 0)
        policy_score = policy_results['summary'].get('compliance_score', 0)
        overall_score = (reg_score + policy_score) / 2
        
        if overall_score >= 80:
            status = "PASS"
        elif overall_score >= 60:
            status = "WARNING"
        else:
            status = "FAIL"
        
        report = {
            "report_date": datetime.now().isoformat(),
            "overall_status": status,
            "overall_compliance_score": round(overall_score, 2),
            "regulation_compliance": {
                "score": reg_score,
                "total_violations": regulation_results['summary'].get('total_violations', 0),
                "violations": regulation_results.get('violations', [])
            },
            "policy_compliance": {
                "score": policy_score,
                "total_violations": policy_results['summary'].get('total_violations', 0),
                "violations": policy_results.get('violations', [])
            },
            "detailed_findings": {
                "regulation_matches": len(regulation_results.get('matches', [])),
                "regulation_weak_matches": len(regulation_results.get('weak_matches', [])),
                "policy_matches": len(policy_results.get('matches', [])),
                "policy_weak_matches": len(policy_results.get('weak_matches', []))
            }
        }
        
        return report
    
    def print_report(self, report: Dict):
        """Print compliance report in readable format"""
        print(f"\n{'='*60}")
        print("📊 COMPLIANCE REPORT")
        print(f"{'='*60}\n")
        
        print(f"Overall Status: {report['overall_status']}")
        print(f"Overall Score: {report['overall_compliance_score']}%\n")
        
        print(f"🏛️  CBU Regulations Compliance: {report['regulation_compliance']['score']}%")
        print(f"   Violations found: {report['regulation_compliance']['total_violations']}\n")
        
        print(f"🏦 Bank Policies Compliance: {report['policy_compliance']['score']}%")
        print(f"   Violations found: {report['policy_compliance']['total_violations']}\n")
        
        # Show violations
        if report['regulation_compliance']['violations']:
            print(f"❌ REGULATION VIOLATIONS:")
            for i, violation in enumerate(report['regulation_compliance']['violations'][:5], 1):
                print(f"\n  {i}. Issue: {violation.get('potential_issue', 'Semantic mismatch')}")
                print(f"     Uploaded text: {violation['uploaded_text']}")
                print(f"     Expected (from {violation['regulation_source']}): {violation['matched_regulation']}")
        
        if report['policy_compliance']['violations']:
            print(f"\n❌ POLICY VIOLATIONS:")
            for i, violation in enumerate(report['policy_compliance']['violations'][:5], 1):
                print(f"\n  {i}. Issue: {violation.get('potential_issue', 'Semantic mismatch')}")
                print(f"     Uploaded text: {violation['uploaded_text']}")
                print(f"     Expected (from {violation['policy_source']}): {violation['matched_policy']}")
        
        print(f"\n{'='*60}\n")


def main():
    """Main execution"""
    print(f"\n{'='*60}")
    print("🔍 COMPLIANCE CHECKER")
    print(f"{'='*60}\n")
    
    # Initialize checker
    checker = ComplianceChecker()
    
    # Get uploaded chunks
    try:
        uploaded_chunks = checker.get_uploaded_chunks()
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return
    
    # Check against regulations
    reg_results = checker.check_against_regulations(uploaded_chunks)
    
    # Check against policies
    policy_results = checker.check_against_policies(uploaded_chunks)
    
    # Generate report
    report = checker.generate_report(reg_results, policy_results)
    
    # Print report
    checker.print_report(report)
    
    # Save to file
    import json
    report_file = f"compliance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Full report saved to: {report_file}")


if __name__ == "__main__":
    main()
