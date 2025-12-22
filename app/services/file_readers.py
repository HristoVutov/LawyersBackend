import aiofiles
from pathlib import Path
import docx
import pypdf
import io


from app.services.ocr_service import OCRService

async def read_text_file(path: Path) -> str:
    async with aiofiles.open(path, mode='r', encoding='utf-8', errors='ignore') as f:
        return await f.read()

def read_docx(path: Path) -> str:
    try:
        doc = docx.Document(path)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception as e:
        return f"Error reading DOCX: {e}"

def read_pdf(path: Path) -> str:
    text = ""
    try:
        with open(path, 'rb') as f:
            reader = pypdf.PdfReader(f)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    except Exception as e:
        print(f"Standard PDF read failed: {e}")
    
    # OCR Fallback if text is too short
    if len(text.strip()) < 50:
        print(f"[Reader] PDF has little text ({len(text)} chars). Triggering OCR for {path.name}...")
        ocr_text = OCRService.get_instance().process_pdf(str(path))
        if ocr_text:
            text = ocr_text
            
    return text

def read_image(path: Path) -> str:
    return OCRService.get_instance().process_image(str(path))

async def read_any_file(path: Path) -> str:
    suffix = path.suffix.lower()
    try:
        if suffix == '.txt':
            return await read_text_file(path)
        elif suffix in ('.docx', '.doc'):
            return read_docx(path)
        elif suffix == '.pdf':
            return read_pdf(path)
        elif suffix in ('.png', '.jpg', '.jpeg', '.bmp', '.tiff'):
            return read_image(path)
        else:
            return f"[Unsupported file type: {suffix}]"
    except Exception as e:
        return f"Error reading file: {str(e)}"
