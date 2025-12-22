import sys
import os
import fitz  # PyMuPDF
from paddleocr import PaddleOCR
import logging
import json
import io

# Force stdout to be utf-8 for Windows compatibility
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Check if running in a virtual environment
if sys.prefix == sys.base_prefix:
    print("Warning: It is recommended to run this script in a virtual environment.", file=sys.stderr)

# Suppress PaddleOCR debug logs
logging.getLogger("ppocr").setLevel(logging.ERROR)

class OCRProcessor:
    def __init__(self, lang='bg'):
        # Initialize PaddleOCR
        # use_textline_orientation=True enables orientation classification (handling rotated text)
        print("Initializing PaddleOCR...", file=sys.stderr)
        self.ocr = PaddleOCR(use_textline_orientation=False, lang=lang)

    def process_file(self, path):
        """
        Process a single file (PDF or Image) and return the text.
        """
        if path.lower().endswith('.pdf'):
            return self._process_pdf(path)
        else:
            return self._process_image(path)

    def _process_image(self, image_path):
        print(f"--- Processing Image: {image_path} ---", file=sys.stderr)
        result = self.ocr.ocr(image_path)
        
        full_text = []
        if result and result[0]:
            # result structure: [[[[x1,y1],[x2,y2]...], ("text", confidence)], ...]
            for line in result[0]:
                text = line[1][0]
                full_text.append(text)
        
        return "\n".join(full_text)

    def _process_pdf(self, pdf_path):
        print(f"--- Processing PDF: {pdf_path} ---", file=sys.stderr)
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"Error opening PDF: {e}", file=sys.stderr)
            return ""

        full_text = []

        for page_num, page in enumerate(doc):
            print(f"Processing Page {page_num + 1}/{len(doc)}...", file=sys.stderr)
            
            # Render page to image suitable for OCR
            # Matrix(2, 2) = 2x zoom for better resolution
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            
            # Save to temporary bytes or file?
            # PaddleOCR accepts numpy array or file path.
            # To avoid extra dependencies like opencv-python just for conversion, 
            # we'll save to a temp file.
            temp_img = f"temp_page_{page_num}.png"
            pix.save(temp_img)

            try:
                result = self.ocr.ocr(temp_img)
                
                page_text = []
                if result:
                    # Handle dict output (e.g. Layout Analysis mode)
                    if isinstance(result[0], dict):
                        # Attempt to find text in 'rec_texts' (plural)
                        text_list = result[0].get('rec_texts', [])
                        
                        # Fallback for 'rec_text' (singular) just in case
                        if not text_list:
                             text_list = result[0].get('rec_text', [])

                        if not text_list and 'res' in result[0]:
                             text_list = [item['text'] for item in result[0]['res']]

                        if text_list:
                            page_text.extend(text_list)
                    
                    # Handle list output (standard OCR mode)
                    elif isinstance(result[0], list):
                        for line in result[0]:
                            if isinstance(line, list) and len(line) >= 2:
                                # line structure: [[x,y..], ("text", score)]
                                page_text.append(line[1][0])

                header = f"\n[Page {page_num + 1}]\n"
                full_text.append(header + "\n".join(page_text))
                
                found_chars = sum(len(t) for t in page_text)
                print(f"  Found {found_chars} characters.", file=sys.stderr)
                
            finally:
                # Cleanup temp file
                if os.path.exists(temp_img):
                    os.remove(temp_img)

        return "\n".join(full_text)

def main():
    if len(sys.argv) < 2:
        print("Usage: python paddle_ocr_script.py <path_to_file_or_folder> [--json]", file=sys.stderr)
        return

    target_path = sys.argv[1]
    
    # Check for --json flag
    use_json = "--json" in sys.argv
    
    processor = OCRProcessor(lang='bg')

    if os.path.isfile(target_path):
        text = processor.process_file(target_path)
        
        if use_json:
            output = {"file": target_path, "text": text}
            print(json.dumps(output, ensure_ascii=False), flush=True)
        else:
            print("\n=== FINAL OUTPUT ===")
            print(text)
            
            # Save output to file only in text mode
            output_file = target_path + "_ocr.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"Saved to: {output_file}", file=sys.stderr)
            
    elif os.path.isdir(target_path):
        print(f"Scanning folder: {target_path}", file=sys.stderr)
        supported_exts = ('.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp')
        
        results = []
        
        for filename in os.listdir(target_path):
            if filename.lower().endswith(supported_exts):
                full_path = os.path.join(target_path, filename)
                text = processor.process_file(full_path)
                
                if use_json:
                    results.append({"file": filename, "text": text})
                else:
                    # Save result to .txt file
                    output_file = full_path + "_ocr.txt"
                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write(text)
                    print(f"Saved text to: {output_file}", file=sys.stderr)
        
        if use_json:
            print(json.dumps(results, ensure_ascii=False))

if __name__ == "__main__":
    main()
