import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
import pytesseract
import io
import os
import re
from typing import List, Dict, Tuple, Optional
import pandas as pd

class LocalPDFToMarkdownWithOCR:
    def __init__(self, tesseract_path: str = None):
        """
        Local PDF processor with OCR text extraction
        
        Args:
            tesseract_path: Path to tesseract executable (if not in PATH)
        """
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # Test if tesseract is available
        try:
            pytesseract.get_tesseract_version()
            print("Tesseract OCR found and ready")
        except Exception as e:
            print(f"Warning: Tesseract not found - {e}")
            print("Install Tesseract OCR for text extraction")
    
    def extract_images_from_pdf(self, pdf_path: str, min_width: int = 100, min_height: int = 100, dpi: float = 200) -> List[Dict]:
        """
        Extract images from PDF with higher DPI for better OCR
        """
        doc = fitz.open(pdf_path)
        images = []
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            
            # Extract embedded images
            image_list = page.get_images()
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    if pix.n - pix.alpha < 4 and pix.width >= min_width and pix.height >= min_height:
                        img_data = pix.tobytes("png")
                        img_array = np.frombuffer(img_data, dtype=np.uint8)
                        img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                        
                        if img_cv is not None:
                            images.append({
                                'page_num': page_num + 1,
                                'type': 'embedded',
                                'index': img_index,
                                'image': img_cv,
                                'width': pix.width,
                                'height': pix.height,
                                'bbox': img[1:5] if len(img) > 4 else None
                            })
                    pix = None
                except Exception as e:
                    print(f"Error extracting embedded image {img_index} from page {page_num + 1}: {e}")
            
            # Render full page at high DPI for better OCR
            try:
                mat = fitz.Matrix(dpi/72, dpi/72)  # Convert DPI to scale factor
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                img_array = np.frombuffer(img_data, dtype=np.uint8)
                img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                if img_cv is not None:
                    images.append({
                        'page_num': page_num + 1,
                        'type': 'full_page',
                        'image': img_cv,
                        'width': pix.width,
                        'height': pix.height
                    })
                pix = None
            except Exception as e:
                print(f"Error rendering page {page_num + 1}: {e}")
        
        doc.close()
        return images
    
    def preprocess_image_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR results
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Denoise
        denoised = cv2.medianBlur(gray, 3)
        
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Sharpen
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # Threshold for clean text
        _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def detect_tables(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect table regions in image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Detect horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Detect vertical lines  
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)
        
        # Combine lines
        table_mask = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
        
        # Find table regions
        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        table_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 200 and h > 100:  # Minimum table size
                table_regions.append((x, y, x + w, y + h))
        
        return table_regions
    
    def extract_text_with_layout(self, image: np.ndarray) -> Dict:
        """
        Extract text with position information for better formatting
        """
        processed = self.preprocess_image_for_ocr(image)
        
        try:
            # Get detailed OCR data with bounding boxes
            data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT, 
                                           config='--psm 6')  # PSM 6: uniform block of text
            
            # Filter out low confidence detections
            filtered_data = []
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 30:  # Confidence threshold
                    text = data['text'][i].strip()
                    if text:
                        filtered_data.append({
                            'text': text,
                            'left': data['left'][i],
                            'top': data['top'][i],
                            'width': data['width'][i],
                            'height': data['height'][i],
                            'conf': data['conf'][i],
                            'block_num': data['block_num'][i],
                            'par_num': data['par_num'][i],
                            'line_num': data['line_num'][i],
                            'word_num': data['word_num'][i]
                        })
            
            # Also get simple text extraction as fallback
            simple_text = pytesseract.image_to_string(processed, config='--psm 6').strip()
            
            return {
                'detailed': filtered_data,
                'simple': simple_text,
                'success': True
            }
            
        except Exception as e:
            print(f"OCR Error: {e}")
            return {
                'detailed': [],
                'simple': '',
                'success': False,
                'error': str(e)
            }
    
    def extract_table_from_region(self, image: np.ndarray, region: Tuple[int, int, int, int]) -> str:
        """
        Extract table from specific region and format as markdown
        """
        x1, y1, x2, y2 = region
        table_img = image[y1:y2, x1:x2]
        
        # Preprocess for table OCR
        processed = self.preprocess_image_for_ocr(table_img)
        
        try:
            # Try structured table extraction
            data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT)
            
            # Group text by rows (similar Y coordinates)
            rows = {}
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 30:
                    text = data['text'][i].strip()
                    if text:
                        top = data['top'][i]
                        left = data['left'][i]
                        
                        # Group by rows (allow some Y variance)
                        row_key = round(top / 20) * 20  # Group within 20 pixel ranges
                        if row_key not in rows:
                            rows[row_key] = []
                        rows[row_key].append((left, text))
            
            # Sort rows and create table
            if len(rows) > 1:
                sorted_rows = sorted(rows.items())
                markdown_rows = []
                
                for row_y, cells in sorted_rows:
                    # Sort cells by X position
                    cells.sort(key=lambda x: x[0])
                    cell_texts = [cell[1] for cell in cells]
                    row_text = " | ".join(cell_texts)
                    markdown_rows.append(f"| {row_text} |")
                
                # Add header separator after first row
                if markdown_rows:
                    num_cols = markdown_rows[0].count("|") - 1
                    header_sep = "|" + " --- |" * num_cols
                    if len(markdown_rows) > 1:
                        markdown_rows.insert(1, header_sep)
                
                return "\n".join(markdown_rows)
            
            # Fallback to simple OCR
            else:
                simple_text = pytesseract.image_to_string(processed)
                if simple_text.strip():
                    lines = [line.strip() for line in simple_text.split('\n') if line.strip()]
                    if len(lines) > 1:
                        # Try to format as simple table
                        formatted_lines = []
                        for i, line in enumerate(lines):
                            # Split on multiple spaces (common in OCR output)
                            cells = re.split(r'\s{3,}', line)
                            if len(cells) > 1:
                                row = " | ".join(cells)
                                formatted_lines.append(f"| {row} |")
                            else:
                                formatted_lines.append(f"| {line} |")
                        
                        # Add header separator after first row
                        if len(formatted_lines) > 1 and '|' in formatted_lines[0]:
                            num_cols = formatted_lines[0].count("|") - 1
                            header_sep = "|" + " --- |" * num_cols
                            formatted_lines.insert(1, header_sep)
                        
                        return "\n".join(formatted_lines)
                    else:
                        return f"```\n{simple_text}\n```"
                
                return "*[Table detected but text extraction failed]*"
                
        except Exception as e:
            return f"*[Table extraction error: {str(e)}]*"
    
    def format_text_to_markdown(self, ocr_result: Dict, image_type: str) -> str:
        """
        Format extracted text into proper markdown
        """
        if not ocr_result['success'] or not ocr_result['simple']:
            return "*[No text detected or OCR failed]*"
        
        text = ocr_result['simple']
        lines = text.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                formatted_lines.append("")
                continue
            
            # Heuristics for markdown formatting
            # Headers (short lines, all caps, or ending with colon)
            if len(line) < 60 and (line.isupper() or line.endswith(':')):
                if len(line) < 30:
                    formatted_lines.append(f"## {line}")
                else:
                    formatted_lines.append(f"### {line}")
            
            # Lists (starting with bullet points or numbers)
            elif re.match(r'^[•\-\*]\s+', line) or re.match(r'^\d+[\.\)]\s+', line):
                formatted_lines.append(f"- {line}")
            
            # Regular text
            else:
                formatted_lines.append(line)
        
        return "\n".join(formatted_lines)
    
    def create_markdown_from_image(self, image: np.ndarray, page_num: int, img_type: str, index: int = None) -> str:
        """
        Create comprehensive markdown from image with OCR
        """
        markdown_lines = []
        
        # Header
        if img_type == 'embedded':
            markdown_lines.append(f"## Page {page_num} - Embedded Image {index}")
        else:
            markdown_lines.append(f"## Page {page_num} - Full Page Content")
        
        height, width = image.shape[:2]
        markdown_lines.append(f"*Image dimensions: {width} x {height} pixels*")
        markdown_lines.append("")
        
        # Detect tables first
        table_regions = self.detect_tables(image)
        
        if table_regions:
            markdown_lines.append(f"### Tables Detected ({len(table_regions)} found)")
            markdown_lines.append("")
            
            for i, region in enumerate(table_regions):
                markdown_lines.append(f"#### Table {i + 1}")
                table_markdown = self.extract_table_from_region(image, region)
                markdown_lines.append(table_markdown)
                markdown_lines.append("")
        
        # Extract all text
        print(f"  Extracting text from {img_type} on page {page_num}...")
        ocr_result = self.extract_text_with_layout(image)
        
        if ocr_result['success'] and ocr_result['simple']:
            if not table_regions:  # If no tables, format as general content
                markdown_lines.append("### Extracted Content")
                markdown_lines.append("")
                formatted_text = self.format_text_to_markdown(ocr_result, "general")
                markdown_lines.append(formatted_text)
            else:
                # Add any text not captured in tables
                markdown_lines.append("### Additional Text Content")
                markdown_lines.append("")
                markdown_lines.append("```")
                markdown_lines.append(ocr_result['simple'])
                markdown_lines.append("```")
        else:
            markdown_lines.append("### Text Extraction")
            if 'error' in ocr_result:
                markdown_lines.append(f"*Error: {ocr_result['error']}*")
            else:
                markdown_lines.append("*No text detected in this image*")
        
        markdown_lines.append("")
        markdown_lines.append("---")
        markdown_lines.append("")
        
        return "\n".join(markdown_lines)
    
    def process_pdf_to_markdown(self, 
                               pdf_path: str, 
                               output_file: str = None,
                               include_embedded: bool = True,
                               include_full_pages: bool = True,
                               min_image_size: int = 100,
                               dpi: float = 200) -> str:
        """
        Process PDF with full text extraction to markdown
        
        Args:
            pdf_path: Path to PDF file
            output_file: Output markdown file path
            include_embedded: Include embedded images
            include_full_pages: Include full page renders
            min_image_size: Minimum image dimension to process
            dpi: DPI for page rendering (higher = better quality, slower)
        """
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        print(f"Processing PDF: {pdf_path}")
        print(f"Using DPI: {dpi} (higher DPI = better OCR quality)")
        
        images = self.extract_images_from_pdf(pdf_path, min_image_size, min_image_size, dpi)
        
        # Filter based on preferences
        if not include_embedded:
            images = [img for img in images if img['type'] != 'embedded']
        if not include_full_pages:
            images = [img for img in images if img['type'] != 'full_page']
        
        print(f"Found {len(images)} images to process")
        
        markdown_sections = []
        
        # Add document header
        markdown_sections.append(f"# PDF Content Extraction: {os.path.basename(pdf_path)}")
        markdown_sections.append(f"*Generated with OCR text extraction*")
        markdown_sections.append("")
        markdown_sections.append("---")
        markdown_sections.append("")
        
        processed = 0
        total_text_extracted = 0
        
        for img_info in images:
            try:
                page_num = img_info['page_num']
                img_type = img_info['type']
                image = img_info['image']
                index = img_info.get('index')
                
                print(f"Processing {img_type} from page {page_num} ({img_info['width']}x{img_info['height']})")
                
                # Generate markdown with OCR
                markdown = self.create_markdown_from_image(image, page_num, img_type, index)
                
                markdown_sections.append(markdown)
                processed += 1
                
                # Count extracted text
                if "Extracted Content" in markdown or "Table" in markdown:
                    total_text_extracted += 1
                
            except Exception as e:
                print(f"Error processing image from page {img_info['page_num']}: {e}")
                continue
        
        print(f"Successfully processed {processed}/{len(images)} images")
        print(f"Text extracted from {total_text_extracted} images")
        
        # Combine all sections
        final_markdown = "\n".join(markdown_sections)
        
        # Add processing summary
        final_markdown += f"\n\n## Processing Summary\n\n"
        final_markdown += f"- **Total images processed:** {processed}\n"
        final_markdown += f"- **Images with text extracted:** {total_text_extracted}\n"
        final_markdown += f"- **DPI used:** {dpi}\n"
        final_markdown += f"- **Source PDF:** {pdf_path}\n"
        final_markdown += f"- **Generated:** {os.path.basename(output_file) if output_file else 'In-memory'}\n"
        
        # Save to file
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(final_markdown)
                print(f"\nMarkdown saved to: {output_file}")
            except Exception as e:
                print(f"Error saving file: {e}")
                return final_markdown
        
        return final_markdown
    
    def process_specific_pages(self, pdf_path: str, pages: List[int], output_file: str = None, dpi: float = 200) -> str:
        """
        Process only specific pages with OCR
        """
        doc = fitz.open(pdf_path)
        markdown_sections = []
        
        markdown_sections.append(f"# PDF Content Extraction (Pages {', '.join(map(str, pages))})")
        markdown_sections.append(f"*Source: {os.path.basename(pdf_path)}*")
        markdown_sections.append("")
        markdown_sections.append("---")
        markdown_sections.append("")
        
        for page_num in pages:
            if page_num > doc.page_count:
                print(f"Page {page_num} does not exist (PDF has {doc.page_count} pages)")
                continue
                
            page = doc[page_num - 1]  # Convert to 0-indexed
            
            # Render page as high-quality image
            mat = fitz.Matrix(dpi/72, dpi/72)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            img_array = np.frombuffer(img_data, dtype=np.uint8)
            img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            pix = None
            
            if img_cv is not None:
                print(f"Processing page {page_num} with OCR...")
                markdown = self.create_markdown_from_image(img_cv, page_num, 'full_page')
                markdown_sections.append(markdown)
        
        doc.close()
        
        final_markdown = "\n".join(markdown_sections)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(final_markdown)
            print(f"Markdown saved to: {output_file}")
        
        return final_markdown

# Usage example
if __name__ == "__main__":
    # Initialize processor
    # On Windows, you might need to specify tesseract path:
    # processor = LocalPDFToMarkdownWithOCR(tesseract_path=r'C:\Program Files\Tesseract-OCR\tesseract.exe')
    processor = LocalPDFToMarkdownWithOCR()
    
    # Example usage
    pdf_path = "tomei_original.pdf"  # Replace with your PDF path
    output_md = "extracted_content_with_text.md"
    
    try:
        # Process entire PDF with OCR
        result = processor.process_pdf_to_markdown(
            pdf_path=pdf_path,
            output_file=output_md,
            include_embedded=True,
            include_full_pages=True,
            min_image_size=50,
            dpi=200  # Higher DPI = better OCR quality but slower
        )
        
        print("\n" + "="*60)
        print("PROCESSING COMPLETE WITH TEXT EXTRACTION!")
        print("="*60)
        print(f"Output saved to: {output_md}")
        
        # Example: Process specific pages only
        # processor.process_specific_pages(pdf_path, [1, 2, 3], "pages_1_2_3.md")
        
    except FileNotFoundError:
        print("Error: PDF file not found. Please check the path.")
        print("Also make sure Tesseract OCR is installed for text extraction.")
    except Exception as e:
        print(f"Error processing PDF: {e}")