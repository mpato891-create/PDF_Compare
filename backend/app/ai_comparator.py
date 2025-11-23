# backend/app/ai_comparator.py
# Smart Contract Comparator - Enhanced

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Tuple, Optional
from difflib import SequenceMatcher

# ============ Clause Representation ============

class ClauseDiffType(Enum):
    MISSING_IN_OTHER = "missing_in_other"
    EXTRA_IN_OTHER = "extra_in_other"
    MODIFIED = "modified"
    SAME = "same"

@dataclass
class Clause:
    """Represents a legal clause"""
    number: str
    text: str
    
    def get_normalized_text(self) -> str:
        """Normalize text for comparison (Critical for SequenceMatcher accuracy)"""
        # Convert to lowercase and remove extra spaces
        text = self.text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        
        # Remove non-alphanumeric characters (keep Arabic letters)
        # This regex keeps Arabic chars, English chars, and numbers
        text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)
        
        # Arabic-specific normalization (unify Alef forms, Teh Marbuta, etc.)
        # This significantly improves matching accuracy for Arabic texts
        text = re.sub(r'[أإآ]', 'ا', text)
        text = re.sub(r'ة', 'ه', text)
        text = re.sub(r'ى', 'ي', text)
        
        return text.strip()

@dataclass
class ClauseDiff:
    """Difference between two clauses"""
    diff_type: ClauseDiffType
    standard_clause: Optional[Clause]
    other_clause: Optional[Clause]
    similarity_score: float = 0.0
    reason: str = ""

# ============ Clause Extraction ============

class SmartClauseExtractor:
    """Smart extraction of legal clauses"""
    
    def __init__(self):
        # Improved regex to catch "Article", "Clause", "Section" in English and Arabic
        # Matches: "Article 1", "Clause 1.1", "المادة 1", "البند الأول", etc.
        self.CLAUSE_PATTERN = r'(ARTICLE|SECTION|CHAPTER|CLAUSE|البند|المادة)\s+([(\d\u0660-\u0669\w\.]{1,10})'
        
        self.IGNORE_PATTERNS = [
            r'(Table|Contents|Signature|Witness|Date|Page|محضر|إنه في يوم|العنوان|صفحة)',
            r'(First Party|Second Party|Signed|الطرف الأول|الطرف الثاني|التوقيع)',
        ]
    
    def extract_clauses(self, text: str) -> List[Clause]:
        """Extract clauses from text using regex, with paragraph fallback"""
        clauses = []
        
        # 1. Try Regex Extraction first
        pattern = re.compile(self.CLAUSE_PATTERN, re.IGNORECASE)
        last_end = -1
        
        for match in pattern.finditer(text):
            if match.start() < last_end:
                continue
                
            clause_indicator = match.group(1).strip()
            clause_number_raw = match.group(2).strip()
            clause_start = match.start()
            
            # Find the start of the next clause to determine the end of current one
            next_match = pattern.search(text, clause_start + len(clause_indicator) + len(clause_number_raw) + 1)
            if next_match:
                clause_end = next_match.start()
            else:
                clause_end = len(text)
            
            clause_full_text = text[clause_start:clause_end].strip()
            
            # Remove the header (e.g., "Article 1") to get the body text
            clause_text = re.sub(r'^{}\s+{}'.format(re.escape(clause_indicator), re.escape(clause_number_raw)), '', clause_full_text, 1, re.MULTILINE).strip()
            
            # Clean and store
            if clause_text and len(clause_text) > 20: # Minimum length filter
                clause_text = re.sub(r'\n+', ' ', clause_text) # Flatten newlines
                
                is_ignored = any(re.search(p, clause_full_text, re.IGNORECASE) for p in self.IGNORE_PATTERNS)
                if not is_ignored:
                    clauses.append(Clause(
                        number=f"{clause_indicator} {clause_number_raw}",
                        text=clause_text
                    ))
                    last_end = clause_end
        
        # 2. Fallback Mechanism: Paragraph Splitting
        # If regex didn't find enough clauses (e.g., unstructured text), split by double newlines
        if len(clauses) < 3:
            # print("⚠️ Regex extraction found few clauses. Switching to paragraph mode.")
            paragraphs = re.split(r'\n\s*\n', text)
            idx = 1
            for para in paragraphs:
                para = para.strip()
                if len(para) > 50: # Only consider substantial paragraphs
                    # Check ignore patterns again
                    is_ignored = any(re.search(p, para, re.IGNORECASE) for p in self.IGNORE_PATTERNS)
                    if not is_ignored:
                        clauses.append(Clause(
                            number=f"Para {idx}",
                            text=para
                        ))
                        idx += 1
                        
        return clauses

# ============ Comparison Logic ============

class SmartContractComparator:
    """Smart Contract Comparator Logic"""
    
    def __init__(self, similarity_threshold: float = 0.85): # Increased threshold slightly for better precision
        self.extractor = SmartClauseExtractor()
        self.similarity_threshold = similarity_threshold
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity ratio between two texts"""
        return SequenceMatcher(None, text1, text2).ratio()
    
    def compare_documents(self, standard_text: str, other_text: str) -> Dict:
        """Perform full comparison between two documents"""
        standard_clauses = self.extractor.extract_clauses(standard_text)
        other_clauses = self.extractor.extract_clauses(other_text)
        
        # Build a map for faster lookup
        other_by_number = {c.number: c for c in other_clauses}
        other_matched = set()
        
        differences = []
        
        # 1. Check Standard Clauses against Other
        for std_clause in standard_clauses:
            # Try to find by explicit number match first
            if std_clause.number in other_by_number:
                other_clause = other_by_number[std_clause.number]
                other_matched.add(std_clause.number)
                
                similarity = self._calculate_similarity(
                    std_clause.get_normalized_text(),
                    other_clause.get_normalized_text()
                )
                
                if similarity < self.similarity_threshold:
                    differences.append(ClauseDiff(
                        diff_type=ClauseDiffType.MODIFIED,
                        standard_clause=std_clause,
                        other_clause=other_clause,
                        similarity_score=similarity,
                        reason=f"Modified: {similarity*100:.1f}% similarity"
                    ))
            else:
                # If number not found, try to find by content similarity (fuzzy search)
                # This handles cases where clause numbers might have shifted
                best_match = None
                best_score = 0.0
                
                for other_c in other_clauses:
                    if other_c.number in other_matched: continue
                    
                    score = self._calculate_similarity(std_clause.get_normalized_text(), other_c.get_normalized_text())
                    if score > best_score:
                        best_score = score
                        best_match = other_c
                
                if best_match and best_score > 0.70: # Found a content match despite number mismatch
                     other_matched.add(best_match.number)
                     if best_score < self.similarity_threshold:
                        differences.append(ClauseDiff(
                            diff_type=ClauseDiffType.MODIFIED,
                            standard_clause=std_clause,
                            other_clause=best_match,
                            similarity_score=best_score,
                            reason=f"Content match found (Numbers differ: {std_clause.number} vs {best_match.number})"
                        ))
                else:
                    differences.append(ClauseDiff(
                        diff_type=ClauseDiffType.MISSING_IN_OTHER,
                        standard_clause=std_clause,
                        other_clause=None,
                        reason="Missing in comparison document"
                    ))
        
        # 2. Check for Extra Clauses in Other
        for other_clause in other_clauses:
            if other_clause.number not in other_matched:
                # Double check to ensure it wasn't matched fuzzily above
                is_actually_extra = True
                
                # (Optional: Add reverse fuzzy check here if needed)
                
                if is_actually_extra:
                    differences.append(ClauseDiff(
                        diff_type=ClauseDiffType.EXTRA_IN_OTHER,
                        standard_clause=None,
                        other_clause=other_clause,
                        reason="Extra clause found"
                    ))
        
        return {
            "total_standard": len(standard_clauses),
            "total_other": len(other_clauses),
            "differences_count": len(differences),
            "differences": differences,
            "standard_clauses": standard_clauses,
            "other_clauses": other_clauses
        }

def compare_with_ai(standard_text: str, other_text: str) -> str:
    """Generate a smart comparison report"""
    
    comparator = SmartContractComparator(similarity_threshold=0.85)
    result = comparator.compare_documents(standard_text, other_text)
    
    # Build the report
    report = []
    report.append("# 📋 Smart Contract Comparison Report")
    report.append("")
    
    # Summary
    report.append("## 📊 Summary")
    report.append(f"- Clauses in Standard Doc: **{result['total_standard']}**")
    report.append(f"- Clauses in Comparison Doc: **{result['total_other']}**")
    report.append(f"- Differences Detected: **{result['differences_count']}**")
    report.append("")
    
    if not result['differences']:
        report.append("✅ **Documents are identical!**")
        return "\n".join(report)
    
    # Detailed Differences
    report.append("## 🔴 Detected Differences:")
    report.append("")
    
    idx = 1
    preview_length = 200
    
    for diff in result['differences']:
        if diff.diff_type == ClauseDiffType.MISSING_IN_OTHER:
            report.append(f"### {idx}. ❌ MISSING - ({diff.standard_clause.number})")
            report.append(f"**Status:** Present in Standard, Missing in Comparison")
            report.append(f"**Text:** _{diff.standard_clause.text[:preview_length]}..._")
            report.append("")
            idx += 1
        
        elif diff.diff_type == ClauseDiffType.EXTRA_IN_OTHER:
            report.append(f"### {idx}. ➕ EXTRA - ({diff.other_clause.number})")
            report.append(f"**Status:** Present in Comparison, Missing in Standard")
            report.append(f"**Text:** _{diff.other_clause.text[:preview_length]}..._")
            report.append("")
            idx += 1
        
        elif diff.diff_type == ClauseDiffType.MODIFIED:
            report.append(f"### {idx}. 🔄 MODIFIED - ({diff.standard_clause.number})")
            report.append(f"**Similarity Score:** {diff.similarity_score*100:.1f}%")
            report.append(f"**Standard Version:**")
            report.append(f"> {diff.standard_clause.text[:preview_length]}...")
            report.append(f"**Comparison Version:**")
            report.append(f"> {diff.other_clause.text[:preview_length]}...")
            report.append("")
            idx += 1
    
    return "\n".join(report)