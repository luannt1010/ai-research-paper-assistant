import re
import unicodedata
import traceback
from tqdm import tqdm
from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

def clean_english_text(text: str) -> str:
    # normalize unicode
    text = unicodedata.normalize("NFC", text)
    # remove control characters ex: \0x88...
    text = "".join(
        char for char in text
        if not unicodedata.category(char).startswith("C")
        or char in "\n\t"
    )
    # remove tab
    text = re.sub(r"[ \t]+", " ", text)
    # Remove spaces around newlines
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    # Keep at most one blank line between paragraphs
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()

class SimpleLoader:

    def load_pdf(self, pdf_file: str) -> List[Document]:
        path = Path(pdf_file)
        docs = PyPDFLoader(pdf_file, extract_images=True).load()
        for doc in docs:
            doc.metadata.update({
                "filename": path.name,
                "document_id": path.stem
            })
            doc.page_content = clean_english_text(doc.page_content)
        return docs

    def load_dir(self, dir_path: str) -> List[Document]:
        directory = Path(dir_path)
        if not directory.exists():
            raise FileExistsError(f"Directory not found: {directory}")
        pdf_files = directory.glob("*.pdf")
        if not pdf_files:
            raise ValueError(f"No PDF files found in {dir_path}")

        all_docs = []
        for pdf_file in tqdm(pdf_files, desc="Loading PDFs"):
            try:
                all_docs.extend(self.load_pdf(pdf_file))
            except Exception as e:
                print(f"\nERROR FILE: {pdf_file}")
                traceback.print_exc()
                print(e)
        return all_docs

