import re
import pandas as pd
from typing import Dict, List, Tuple, Any
import json

class FinancialDataDetector:
    """Enhanced financial data detection and table formatting system"""
    
    def __init__(self):
        # Financial keywords and patterns
        self.financial_keywords = [
            'revenue', 'profit', 'loss', 'margin', 'debt', 'equity', 'assets', 'liabilities',
            'cash', 'opbdit', 'ebitda', 'earnings', 'dividend', 'capex', 'operating',
            'net income', 'gross profit', 'total assets', 'total debt', 'gearing ratio',
            'coverage ratio', 'return on', 'interest', 'tax', 'depreciation', 'amortization'
        ]
        
        # Currency patterns
        self.currency_patterns = [
            r'RM\s*\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|mil|billion|bil|thousand|k))?',
            r'\$\s*\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|mil|billion|bil|thousand|k))?',
            r'USD\s*\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|mil|billion|bil|thousand|k))?',
            r'\d+(?:,\d{3})*(?:\.\d+)?\s*(?:million|mil|billion|bil|thousand|k)',
        ]
        
        # Ratio and percentage patterns
        self.ratio_patterns = [
            r'\d+(?:\.\d+)?\s*times',
            r'\d+(?:\.\d+)?\s*%',
            r'\d+(?:\.\d+)?\s*percent',
            r'ratio\s*:\s*\d+(?:\.\d+)?',
        ]
        
        # Date patterns for financial periods
        self.date_patterns = [
            r'FY\s*(?:Dec|December)\s*\d{4}',
            r'Q[1-4]\s*\d{4}',
            r'\d{4}',
            r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s*\d{4}',
        ]

    def detect_financial_numbers(self, text: str) -> List[Dict[str, Any]]:
        """Detect financial numbers and their context in text"""
        financial_data = []
        
        # Split text into sentences for better context
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check if sentence contains financial keywords
            has_financial_keyword = any(keyword.lower() in sentence.lower() 
                                     for keyword in self.financial_keywords)
            
            if has_financial_keyword:
                # Extract currency values
                for pattern in self.currency_patterns:
                    matches = re.finditer(pattern, sentence, re.IGNORECASE)
                    for match in matches:
                        financial_data.append({
                            'type': 'currency',
                            'value': match.group(),
                            'context': sentence,
                            'position': match.span()
                        })
                
                # Extract ratios and percentages
                for pattern in self.ratio_patterns:
                    matches = re.finditer(pattern, sentence, re.IGNORECASE)
                    for match in matches:
                        financial_data.append({
                            'type': 'ratio',
                            'value': match.group(),
                            'context': sentence,
                            'position': match.span()
                        })
        
        return financial_data

    def extract_financial_tables(self, text: str) -> List[Dict[str, Any]]:
        """Extract structured financial data that can be formatted as tables"""
        tables = []
        
        # Pattern for key-value pairs (e.g., "Revenue: RM 4,538.9 million")
        kv_pattern = r'([A-Za-z\s]+?):\s*((?:RM|USD|\$)?\s*\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|mil|billion|bil|thousand|k|times|%))?)'
        kv_matches = re.findall(kv_pattern, text, re.IGNORECASE)
        
        if len(kv_matches) >= 3:  # At least 3 key-value pairs to form a table
            table_data = []
            for key, value in kv_matches:
                table_data.append({
                    'metric': key.strip(),
                    'value': value.strip()
                })
            
            tables.append({
                'type': 'key_value_table',
                'title': 'Financial Metrics',
                'data': table_data
            })
        
        # Pattern for comparative data (e.g., "2019: RM 4,538.9 million, 2018: RM 4,353.6 million")
        comparative_pattern = r'(\d{4})[:\s]*([^,\n]+?)(?:,|\n|$)'
        comparative_matches = re.findall(comparative_pattern, text)
        
        if len(comparative_matches) >= 2:
            table_data = []
            for year, value in comparative_matches:
                # Clean up the value
                value = re.sub(r'[^\w\s\.\,\$\%]', '', value).strip()
                if value:
                    table_data.append({
                        'year': year,
                        'value': value
                    })
            
            if table_data:
                tables.append({
                    'type': 'time_series_table',
                    'title': 'Year-over-Year Comparison',
                    'data': table_data
                })
        
        return tables

    def format_as_markdown_table(self, table_data: Dict[str, Any]) -> str:
        """Format extracted data as a markdown table"""
        if table_data['type'] == 'key_value_table':
            markdown = f"### {table_data['title']}\n\n"
            markdown += "| Metric | Value |\n"
            markdown += "|--------|-------|\n"
            
            for row in table_data['data']:
                metric = row['metric'].replace('|', '\\|')
                value = row['value'].replace('|', '\\|')
                markdown += f"| {metric} | {value} |\n"
            
            return markdown
        
        elif table_data['type'] == 'time_series_table':
            markdown = f"### {table_data['title']}\n\n"
            markdown += "| Year | Value |\n"
            markdown += "|------|-------|\n"
            
            for row in table_data['data']:
                year = row['year'].replace('|', '\\|')
                value = row['value'].replace('|', '\\|')
                markdown += f"| {year} | {value} |\n"
            
            return markdown
        
        return ""

    def enhance_response_with_tables(self, response_text: str, context_chunks: List[str]) -> str:
        """Enhance response with automatically detected financial tables"""
        enhanced_response = response_text
        
        # Combine all context for comprehensive analysis
        full_context = " ".join(context_chunks)
        
        # Extract financial tables from context
        tables = self.extract_financial_tables(full_context)
        
        # If tables are found, append them to the response
        if tables:
            enhanced_response += "\n\n## Financial Data Summary\n\n"
            for table in tables:
                table_markdown = self.format_as_markdown_table(table)
                if table_markdown:
                    enhanced_response += table_markdown + "\n\n"
        
        return enhanced_response

    def detect_financial_query_type(self, query: str) -> str:
        """Detect the type of financial query to determine response format"""
        query_lower = query.lower()
        
        # Simple identification queries
        simple_patterns = [
            r'what is the .+?(?:\?|$)',
            r'how much .+?(?:\?|$)',
            r'when .+?(?:\?|$)',
            r'where .+?(?:\?|$)',
            r'who .+?(?:\?|$)',
        ]
        
        for pattern in simple_patterns:
            if re.search(pattern, query_lower):
                return 'simple'
        
        # Complex analysis queries
        complex_keywords = [
            'analyze', 'compare', 'trend', 'performance', 'breakdown', 'summary',
            'overview', 'analysis', 'evaluation', 'assessment', 'review'
        ]
        
        if any(keyword in query_lower for keyword in complex_keywords):
            return 'complex'
        
        # Financial data queries
        financial_keywords = [
            'revenue', 'profit', 'financial', 'earnings', 'debt', 'ratio',
            'margin', 'performance', 'metrics', 'results'
        ]
        
        if any(keyword in query_lower for keyword in financial_keywords):
            return 'financial'
        
        return 'general'

# Example usage and testing
if __name__ == "__main__":
    detector = FinancialDataDetector()
    
    # Test with sample text
    sample_text = """
    AEON (M)'s revenue increased to RM 4,538.9 million in 2019 from RM 4,353.6 million in 2018.
    The company's OPBDIT margin improved to 18.60% in 2019 compared to 17.06% in 2018.
    Total debt increased significantly to RM 3,074.7 million in 2019 from RM 1,046.2 million in 2018.
    The adjusted gearing ratio rose to 1.47 times in 2019 from 0.97 times in 2018.
    """
    
    # Test financial number detection
    financial_numbers = detector.detect_financial_numbers(sample_text)
    print("Detected Financial Numbers:")
    for item in financial_numbers:
        print(f"  {item['type']}: {item['value']} (Context: {item['context'][:50]}...)")
    
    # Test table extraction
    tables = detector.extract_financial_tables(sample_text)
    print("\nExtracted Tables:")
    for table in tables:
        print(f"  {table['type']}: {table['title']}")
        markdown = detector.format_as_markdown_table(table)
        print(markdown)

