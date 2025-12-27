"""
Correction Generator
Takes compliance violations and generates actionable corrections
"""

import re
from typing import Dict, List
import json
from datetime import datetime

class CorrectionGenerator:
    """Generate corrected versions based on violations"""
    
    def __init__(self):
        """Initialize correction patterns"""
        self.number_patterns = {
            'days': r'(\d+)\s*(?:kalendar|календар|ish|иш)?\s*(?:kun|кун|day|days)',
            'percent': r'(\d+(?:\.\d+)?)\s*(?:%|процент|foiz|фоиз)',
            'months': r'(\d+)\s*(?:oy|ой|месяц|month|months)',
            'years': r'(\d+)\s*(?:yil|йил|год|year|years)'
        }
    
    def extract_correct_value(self, violation: Dict) -> Dict:
        """
        Extract the correct value from regulation/policy text
        
        Args:
            violation: Violation dictionary from compliance report
            
        Returns:
            Correction details
        """
        uploaded_text = violation.get('uploaded_text', '')
        matched_text = violation.get('matched_regulation') or violation.get('matched_policy', '')
        potential_issue = violation.get('potential_issue', '')
        
        correction = {
            'section_id': self._extract_section_number(uploaded_text),
            'violation_type': 'regulation' if 'matched_regulation' in violation else 'policy',
            'source_document': violation.get('regulation_source') or violation.get('policy_source'),
            'current_text': uploaded_text,
            'issue_description': potential_issue,
            'corrections': []
        }
        
        # Parse the potential_issue to extract wrong/correct values
        if potential_issue:
            corrections = self._parse_issue_description(potential_issue, uploaded_text, matched_text)
            correction['corrections'] = corrections
        
        return correction
    
    def _extract_section_number(self, text: str) -> str:
        """Extract section number from text (e.g., '7.4.5.3')"""
        match = re.match(r'^(\d+\.(?:\d+\.)*\d*)', text.strip())
        if match:
            return match.group(1)
        return 'unknown'
    
    def _parse_issue_description(self, issue: str, uploaded_text: str, reference_text: str) -> List[Dict]:
        """
        Parse issue description to extract corrections
        
        Args:
            issue: Issue string like "PERCENT: Uploaded says 85.0, reference says 80.0"
            uploaded_text: Original uploaded text
            reference_text: Reference regulation/policy text
            
        Returns:
            List of correction dictionaries
        """
        corrections = []
        
        # Split by | to handle multiple issues
        issues = issue.split('|')
        
        for single_issue in issues:
            single_issue = single_issue.strip()
            
            # Parse pattern: "TYPE: Uploaded says X, reference says Y"
            match = re.match(r'(\w+):\s*Uploaded says ([\d.]+),\s*reference says ([\d.]+)', single_issue)
            
            if match:
                value_type = match.group(1).lower()
                uploaded_value = match.group(2)
                correct_value = match.group(3)
                
                # Skip if values are too different (likely false positive)
                try:
                    if abs(float(uploaded_value) - float(correct_value)) > 100:
                        continue
                except:
                    pass
                
                # Find the actual text in uploaded document
                current_phrase = self._find_phrase_with_value(uploaded_text, uploaded_value, value_type)
                corrected_phrase = self._find_phrase_with_value(reference_text, correct_value, value_type)
                
                correction_item = {
                    'field': value_type,
                    'current_value': uploaded_value,
                    'correct_value': correct_value,
                    'current_phrase': current_phrase or f"{uploaded_value} {self._get_unit_name(value_type)}",
                    'corrected_phrase': corrected_phrase or f"{correct_value} {self._get_unit_name(value_type)}",
                    'severity': self._determine_severity(value_type, uploaded_value, correct_value)
                }
                
                corrections.append(correction_item)
        
        # Remove duplicates
        seen = set()
        unique_corrections = []
        for c in corrections:
            key = (c['field'], c['current_value'], c['correct_value'])
            if key not in seen:
                seen.add(key)
                unique_corrections.append(c)
        
        return unique_corrections
    
    def _find_phrase_with_value(self, text: str, value: str, value_type: str) -> str:
        """Find the phrase containing the specific value"""
        # Create pattern based on value type
        if value_type == 'days':
            pattern = rf'\b{value}\s*(?:kalendar|календар|ish|иш)?\s*(?:kun|кун|day)[^.;]*'
        elif value_type == 'percent':
            pattern = rf'\b{value}\s*(?:%|процент|foiz|фоиз)[^.;]*'
        elif value_type == 'months':
            pattern = rf'\b{value}\s*(?:oy|ой|месяц|month)[^.;]*'
        else:
            pattern = rf'\b{value}\b[^.;]*'
        
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0).strip()
        return None
    
    def _get_unit_name(self, value_type: str) -> str:
        """Get unit name for value type"""
        units = {
            'days': 'kalendar kun',
            'percent': '%',
            'months': 'oy',
            'years': 'yil'
        }
        return units.get(value_type, '')
    
    def _determine_severity(self, value_type: str, current: str, correct: str) -> str:
        """Determine severity of the violation"""
        try:
            diff = abs(float(current) - float(correct))
            
            if value_type == 'percent':
                if diff >= 5:
                    return 'CRITICAL'
                elif diff >= 1:
                    return 'HIGH'
                else:
                    return 'MEDIUM'
            elif value_type == 'days':
                if diff >= 5:
                    return 'HIGH'
                else:
                    return 'MEDIUM'
            else:
                return 'MEDIUM'
        except:
            return 'MEDIUM'
    
    def generate_correction_report(self, compliance_report: Dict) -> Dict:
        """
        Generate actionable correction report
        
        Args:
            compliance_report: Output from compliance_checker
            
        Returns:
            Correction report with actionable fixes
        """
        print(f"\n{'='*60}")
        print("🔧 GENERATING CORRECTION REPORT")
        print(f"{'='*60}\n")
        
        correction_report = {
            'report_date': datetime.now().isoformat(),
            'overall_status': compliance_report['overall_status'],
            'overall_compliance_score': compliance_report['overall_compliance_score'],
            'total_corrections_needed': 0,
            'critical_corrections': [],
            'high_priority_corrections': [],
            'medium_priority_corrections': [],
            'summary': {
                'critical_count': 0,
                'high_count': 0,
                'medium_count': 0
            }
        }
        
        # Process regulation violations
        reg_violations = compliance_report.get('regulation_compliance', {}).get('violations', [])
        for violation in reg_violations:
            correction = self.extract_correct_value(violation)
            self._categorize_correction(correction, correction_report)
        
        # Process policy violations
        policy_violations = compliance_report.get('policy_compliance', {}).get('violations', [])
        for violation in policy_violations:
            correction = self.extract_correct_value(violation)
            self._categorize_correction(correction, correction_report)
        
        # Calculate totals
        correction_report['total_corrections_needed'] = (
            correction_report['summary']['critical_count'] +
            correction_report['summary']['high_count'] +
            correction_report['summary']['medium_count']
        )
        
        return correction_report
    
    def _categorize_correction(self, correction: Dict, report: Dict):
        """Categorize correction by severity"""
        if not correction['corrections']:
            return
        
        for corr in correction['corrections']:
            severity = corr['severity']
            
            correction_entry = {
                'section': correction['section_id'],
                'violation_type': correction['violation_type'],
                'source_document': correction['source_document'],
                'field': corr['field'],
                'current_value': f"{corr['current_value']} {self._get_unit_name(corr['field'])}",
                'correct_value': f"{corr['correct_value']} {self._get_unit_name(corr['field'])}",
                'current_phrase': corr['current_phrase'],
                'corrected_phrase': corr['corrected_phrase'],
                'action_required': f"Change '{corr['current_value']}' to '{corr['correct_value']}'"
            }
            
            if severity == 'CRITICAL':
                report['critical_corrections'].append(correction_entry)
                report['summary']['critical_count'] += 1
            elif severity == 'HIGH':
                report['high_priority_corrections'].append(correction_entry)
                report['summary']['high_count'] += 1
            else:
                report['medium_priority_corrections'].append(correction_entry)
                report['summary']['medium_count'] += 1
    
    def print_correction_report(self, report: Dict):
        """Print correction report in readable format"""
        print(f"\n{'='*60}")
        print("📋 CORRECTION REPORT - ACTIONABLE FIXES")
        print(f"{'='*60}\n")
        
        print(f"Status: {report['overall_status']}")
        print(f"Compliance Score: {report['overall_compliance_score']}%")
        print(f"Total Corrections Needed: {report['total_corrections_needed']}\n")
        
        print(f"Breakdown:")
        print(f"  🔴 Critical: {report['summary']['critical_count']}")
        print(f"  🟠 High Priority: {report['summary']['high_count']}")
        print(f"  🟡 Medium Priority: {report['summary']['medium_count']}")
        
        # Print critical corrections
        if report['critical_corrections']:
            print(f"\n{'='*60}")
            print("🔴 CRITICAL CORRECTIONS (Must Fix Immediately)")
            print(f"{'='*60}\n")
            
            for i, corr in enumerate(report['critical_corrections'], 1):
                print(f"{i}. Section {corr['section']} - {corr['field'].upper()}")
                print(f"   Source: {corr['source_document']} ({corr['violation_type']})")
                print(f"   ❌ Current: {corr['current_value']}")
                print(f"   ✅ Must be: {corr['correct_value']}")
                print(f"   📝 Change from: \"{corr['current_phrase']}\"")
                print(f"   📝 Change to:   \"{corr['corrected_phrase']}\"")
                print(f"   Action: {corr['action_required']}\n")
        
        # Print high priority corrections
        if report['high_priority_corrections']:
            print(f"\n{'='*60}")
            print("🟠 HIGH PRIORITY CORRECTIONS")
            print(f"{'='*60}\n")
            
            for i, corr in enumerate(report['high_priority_corrections'], 1):
                print(f"{i}. Section {corr['section']} - {corr['field'].upper()}")
                print(f"   Source: {corr['source_document']}")
                print(f"   ❌ Current: {corr['current_value']} → ✅ Should be: {corr['correct_value']}")
                print(f"   Action: {corr['action_required']}\n")
        
        # Print medium priority corrections
        if report['medium_priority_corrections']:
            print(f"\n{'='*60}")
            print("🟡 MEDIUM PRIORITY CORRECTIONS")
            print(f"{'='*60}\n")
            
            for i, corr in enumerate(report['medium_priority_corrections'], 1):
                print(f"{i}. Section {corr['section']} - {corr['field'].upper()}")
                print(f"   ❌ Current: {corr['current_value']} → ✅ Should be: {corr['correct_value']}\n")
        
        print(f"{'='*60}\n")


def main():
    """Main execution - enhance existing compliance report"""
    print(f"\n{'='*60}")
    print("🔧 CORRECTION GENERATOR")
    print(f"{'='*60}\n")
    
    # Load the most recent compliance report
    import glob
    import os
    
    report_files = glob.glob("compliance_report_*.json")
    if not report_files:
        print("❌ No compliance reports found!")
        print("   Run compliance_checker.py first")
        return
    
    # Get most recent report
    latest_report = max(report_files, key=os.path.getctime)
    print(f"📂 Loading report: {latest_report}")
    
    with open(latest_report, 'r', encoding='utf-8') as f:
        compliance_report = json.load(f)
    
    # Generate corrections
    generator = CorrectionGenerator()
    correction_report = generator.generate_correction_report(compliance_report)
    
    # Print corrections
    generator.print_correction_report(correction_report)
    
    # Save correction report
    output_file = f"correction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(correction_report, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Correction report saved to: {output_file}")
    print("\n✅ You can now provide this to compliance team for fixes!")


if __name__ == "__main__":
    main()
