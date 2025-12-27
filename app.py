"""
Compliance Checker Web UI
Upload PDF contracts and get instant compliance reports
"""

import streamlit as st
import tempfile
import os
from pathlib import Path
import json

# Import your existing modules
from pdf_processor import PDFProcessor
from compliance_checker import ComplianceChecker
from correction_generator import CorrectionGenerator

# Page config
st.set_page_config(
    page_title="Bank Compliance Checker",
    page_icon="🏦",
    layout="wide"
)

# Title
st.title("🏦 Bank Contract Compliance Checker")
st.markdown("Upload a loan contract PDF to check compliance against CBU regulations and bank policies")

# Sidebar info
with st.sidebar:
    st.header("📊 System Info")
    
    # Check ChromaDB status
    try:
        checker = ComplianceChecker()
        if checker.regulations_collection:
            st.success(f"✅ Regulations: {checker.regulations_collection.count()} items")
        if checker.policies_collection:
            st.success(f"✅ Bank Policies: {checker.policies_collection.count()} items")
    except Exception as e:
        st.error(f"❌ ChromaDB Error: {str(e)}")
    
    st.markdown("---")
    st.markdown("### How it works:")
    st.markdown("""
    1. Upload your contract PDF
    2. System extracts & analyzes text
    3. Compares against:
       - CBU Regulations
       - Anor Bank Policies
    4. Get instant compliance report
    5. See exact corrections needed
    """)

# Main content
uploaded_file = st.file_uploader(
    "📄 Upload Contract PDF",
    type=['pdf'],
    help="Upload a loan contract or policy document to check compliance"
)

if uploaded_file is not None:
    # Show file info
    st.info(f"📄 File: {uploaded_file.name} ({uploaded_file.size} bytes)")
    
    # Create columns for buttons
    col1, col2, col3 = st.columns([1, 1, 3])
    
    with col1:
        check_button = st.button("🔍 Check Compliance", type="primary")
    
    with col2:
        if st.session_state.get('compliance_checked'):
            show_corrections = st.button("🔧 Show Corrections")
        else:
            show_corrections = False
    
    if check_button:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        try:
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Process PDF
            status_text.text("📄 Extracting text from PDF...")
            progress_bar.progress(20)
            
            processor = PDFProcessor()
            chunks_with_embeddings, full_text, doc_id = processor.process_pdf(tmp_path)
            
            st.success(f"✅ Extracted {len(chunks_with_embeddings)} chunks from PDF")
            
            # Step 2: Check compliance
            status_text.text("🔍 Checking against regulations...")
            progress_bar.progress(40)
            
            checker = ComplianceChecker()
            uploaded_chunks = checker.get_uploaded_chunks()
            
            status_text.text("🔍 Comparing with CBU regulations...")
            progress_bar.progress(60)
            reg_results = checker.check_against_regulations(uploaded_chunks)
            
            status_text.text("🔍 Comparing with bank policies...")
            progress_bar.progress(80)
            policy_results = checker.check_against_policies(uploaded_chunks)
            
            # Step 3: Generate report
            status_text.text("📋 Generating compliance report...")
            progress_bar.progress(90)
            
            compliance_report = checker.generate_report(reg_results, policy_results)
            
            # Store in session state
            st.session_state['compliance_report'] = compliance_report
            st.session_state['compliance_checked'] = True
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
            # Display results
            st.markdown("---")
            st.header("📊 Compliance Report")
            
            # Overall status
            status = compliance_report['overall_status']
            score = compliance_report['overall_compliance_score']
            
            # Color-coded status
            if status == "PASS":
                st.success(f"### ✅ Status: {status} ({score}%)")
            elif status == "WARNING":
                st.warning(f"### ⚠️ Status: {status} ({score}%)")
            else:
                st.error(f"### ❌ Status: {status} ({score}%)")
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Overall Score",
                    f"{score}%",
                    delta=None
                )
            
            with col2:
                reg_violations = compliance_report['regulation_compliance']['total_violations']
                st.metric(
                    "Regulation Violations",
                    reg_violations,
                    delta=None,
                    delta_color="inverse"
                )
            
            with col3:
                policy_violations = compliance_report['policy_compliance']['total_violations']
                st.metric(
                    "Policy Violations",
                    policy_violations,
                    delta=None,
                    delta_color="inverse"
                )
            
            # Detailed findings
            st.markdown("---")
            
            # Regulation violations
            if compliance_report['regulation_compliance']['violations']:
                with st.expander("🏛️ CBU Regulation Violations", expanded=True):
                    for i, violation in enumerate(compliance_report['regulation_compliance']['violations'], 1):
                        st.markdown(f"**Violation {i}**")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Your Document:**")
                            st.text_area(
                                f"doc_{i}",
                                violation['uploaded_text'],
                                height=100,
                                label_visibility="collapsed"
                            )
                        
                        with col2:
                            st.markdown("**Regulation Requirement:**")
                            st.text_area(
                                f"reg_{i}",
                                violation['matched_regulation'],
                                height=100,
                                label_visibility="collapsed"
                            )
                        
                        if violation.get('potential_issue'):
                            st.error(f"⚠️ Issue: {violation['potential_issue']}")
                        
                        st.caption(f"Source: {violation['regulation_source']}")
                        st.markdown("---")
            else:
                st.success("✅ No regulation violations found!")
            
            # Policy violations
            if compliance_report['policy_compliance']['violations']:
                with st.expander("🏦 Bank Policy Violations", expanded=True):
                    for i, violation in enumerate(compliance_report['policy_compliance']['violations'], 1):
                        st.markdown(f"**Violation {i}**")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Your Document:**")
                            st.text_area(
                                f"doc_p_{i}",
                                violation['uploaded_text'],
                                height=100,
                                label_visibility="collapsed"
                            )
                        
                        with col2:
                            st.markdown("**Policy Requirement:**")
                            st.text_area(
                                f"pol_{i}",
                                violation['matched_policy'],
                                height=100,
                                label_visibility="collapsed"
                            )
                        
                        if violation.get('potential_issue'):
                            st.error(f"⚠️ Issue: {violation['potential_issue']}")
                        
                        st.caption(f"Source: {violation['policy_source']}")
                        st.markdown("---")
            
            # Download report
            st.markdown("---")
            report_json = json.dumps(compliance_report, ensure_ascii=False, indent=2)
            st.download_button(
                label="📥 Download Full Report (JSON)",
                data=report_json,
                file_name=f"compliance_report_{uploaded_file.name}.json",
                mime="application/json"
            )
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback
            with st.expander("Show error details"):
                st.code(traceback.format_exc())
        
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except:
                pass
    
    # Show corrections if requested
    if show_corrections and st.session_state.get('compliance_report'):
        st.markdown("---")
        st.header("🔧 Actionable Corrections")
        
        try:
            generator = CorrectionGenerator()
            correction_report = generator.generate_correction_report(
                st.session_state['compliance_report']
            )
            
            # Summary
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Corrections", correction_report['total_corrections_needed'])
            with col2:
                st.metric("🔴 Critical", correction_report['summary']['critical_count'])
            with col3:
                st.metric("🟠 High Priority", correction_report['summary']['high_count'])
            with col4:
                st.metric("🟡 Medium Priority", correction_report['summary']['medium_count'])
            
            # Critical corrections
            if correction_report['critical_corrections']:
                with st.expander("🔴 CRITICAL - Must Fix Immediately", expanded=True):
                    for i, corr in enumerate(correction_report['critical_corrections'], 1):
                        st.markdown(f"### {i}. Section {corr['section']} - {corr['field'].upper()}")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**❌ Current:**")
                            st.code(corr['current_value'], language=None)
                            st.text_area(
                                f"curr_{i}",
                                corr['current_phrase'],
                                height=60,
                                label_visibility="collapsed"
                            )
                        
                        with col2:
                            st.markdown("**✅ Must Change To:**")
                            st.code(corr['correct_value'], language=None)
                            st.text_area(
                                f"corr_{i}",
                                corr['corrected_phrase'],
                                height=60,
                                label_visibility="collapsed"
                            )
                        
                        st.info(f"📝 Action: {corr['action_required']}")
                        st.caption(f"Source: {corr['source_document']} ({corr['violation_type']})")
                        st.markdown("---")
            
            # High priority corrections
            if correction_report['high_priority_corrections']:
                with st.expander("🟠 HIGH PRIORITY Corrections"):
                    for i, corr in enumerate(correction_report['high_priority_corrections'], 1):
                        st.markdown(f"**{i}. Section {corr['section']} - {corr['field'].upper()}**")
                        st.markdown(f"❌ Current: `{corr['current_value']}` → ✅ Should be: `{corr['correct_value']}`")
                        st.caption(f"Action: {corr['action_required']}")
                        st.markdown("---")
            
            # Download corrections
            correction_json = json.dumps(correction_report, ensure_ascii=False, indent=2)
            st.download_button(
                label="📥 Download Correction Report (JSON)",
                data=correction_json,
                file_name=f"corrections_{uploaded_file.name}.json",
                mime="application/json"
            )
            
        except Exception as e:
            st.error(f"❌ Error generating corrections: {str(e)}")

else:
    # Show instructions when no file uploaded
    st.info("👆 Upload a PDF file to begin compliance checking")
    
    # Show example
    with st.expander("📋 What this tool checks"):
        st.markdown("""
        ### Regulation Compliance
        - ✅ CBU capital requirements
        - ✅ Loan-to-value ratios
        - ✅ Interest rate limits
        - ✅ Collateral requirements
        
        ### Policy Compliance  
        - ✅ Document submission timelines
        - ✅ Monitoring frequencies
        - ✅ Client notification periods
        - ✅ Mandatory clauses
        
        ### Output
        - 📊 Overall compliance score
        - 🔍 Specific violations with sources
        - 🔧 Exact corrections needed
        - 📥 Downloadable reports
        """)
