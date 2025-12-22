import sys
import os
import fitz  # PyMuPDF
import logging
import importlib

# SHIM: Fix for paddleocr/paddlex incompatibility with langchain >= 0.3.0
# paddlex tries to import 'from langchain.docstore.document import Document'
# but in 0.3.0 it is in 'langchain_core.documents'
# Also tries 'from langchain.text_splitter import ...' which is now 'langchain_text_splitters'
try:
    import langchain
    import langchain_core.documents
    
    # Shim 1: langchain.docstore.document -> langchain_core.documents
    try:
        from langchain.docstore import document
    except ImportError:
        from types import ModuleType
        
        docstore = ModuleType("langchain.docstore")
        doc_module = ModuleType("langchain.docstore.document")
        doc_module.Document = langchain_core.documents.Document
        
        docstore.document = doc_module
        sys.modules["langchain.docstore"] = docstore
        sys.modules["langchain.docstore.document"] = doc_module
    
    # Shim 2: langchain.text_splitter -> langchain_text_splitters
    try:
        from langchain import text_splitter
    except (ImportError, AttributeError):
        try:
            import langchain_text_splitters
            sys.modules["langchain.text_splitter"] = langchain_text_splitters
        except ImportError:
            # If langchain_text_splitters is not installed, create empty module
            from types import ModuleType
            text_splitter_module = ModuleType("langchain.text_splitter")
            sys.modules["langchain.text_splitter"] = text_splitter_module
            print("Warning: langchain_text_splitters not installed, text_splitter shim is empty", file=sys.stderr)
            
except Exception as e:
    print(f"Warning: Failed to shim langchain for paddleocr: {e}", file=sys.stderr)

# Suppress PaddleOCR debug logs
logging.getLogger("ppocr").setLevel(logging.ERROR)

class OCRService:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self, lang='bg'):
        try:
            print("Initializing PaddleOCR...", file=sys.stderr)
            # Lazy import to avoid startup crash if shim fails
            from paddleocr import PaddleOCR
            
            # use_textline_orientation=False is faster if we don't expect rotated text
            self.ocr = PaddleOCR(use_textline_orientation=False, lang=lang)
        except Exception as e:
            print(f"Failed to initialize PaddleOCR: {e}", file=sys.stderr)
            self.ocr = None

    def process_image(self, image_path: str) -> str:
        if not self.ocr:
            return ""
            
        try:
            result = self.ocr.ocr(str(image_path))
            
            full_text = []
            if result and result[0]:
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
                        full_text.extend(text_list)
                
                # Handle list output (standard OCR mode)
                elif isinstance(result[0], list):
                    for line in result[0]:
                        if isinstance(line, list) and len(line) >= 2:
                            # line structure: [[x,y..], ("text", score)]
                            text = line[1][0]
                            full_text.append(text)

            return "\n".join(full_text)
        except Exception as e:
            print(f"OCR Error on image {image_path}: {e}", file=sys.stderr)
            return ""

    def process_pdf(self, pdf_path: str) -> str:
        if not self.ocr:
            return ""
            
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"Error opening PDF for OCR: {e}", file=sys.stderr)
            return ""

        full_text = []

        try:
            for page_num, page in enumerate(doc):
                # Matrix(2, 2) = 2x zoom for better resolution
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                
                # Save to temp file as PaddleOCR expects file path or numpy array
                # Using a temp file is safer for memory with large PDFs
                temp_img = f"temp_ocr_page_{page_num}.png"
                pix.save(temp_img)

                try:
                    page_text = self.process_image(temp_img)
                    header = f"\n[Page {page_num + 1}]\n"
                    full_text.append(header + page_text)
                finally:
                    if os.path.exists(temp_img):
                        os.remove(temp_img)
        except Exception as e:
            print(f"Error processing PDF pages: {e}", file=sys.stderr)
        
        return "\n".join(full_text)
