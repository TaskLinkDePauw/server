import re
import fitz  # PyMuPDF
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt_tab')

def read_and_chunk_pdf_adaptive(pdf_path, max_words=150):
    """
    Reads a PDF and splits its text into chunks around natural sentence boundaries.
    Each chunk tries not to exceed `max_words`.
    """
    doc = fitz.open(pdf_path)
    text_chunks = []
    
    for page_idx, page in enumerate(doc):
        page_text = page.get_text()
        # Clean up whitespace
        page_text = re.sub(r"\s+", " ", page_text).strip()
        
        # Break the page text into sentences
        sentences = sent_tokenize(page_text)
        
        current_chunk = []
        current_length = 0
        
        for sent in sentences:
            sent_word_count = len(sent.split())
            # If adding this sentence exceeds the limit, finalize the current chunk
            if current_length + sent_word_count > max_words and current_chunk:
                chunk_text = " ".join(current_chunk).strip()
                if len(chunk_text) > 30:  # or some minimal length threshold
                    text_chunks.append(chunk_text)
                # Start a new chunk
                current_chunk = [sent]
                current_length = sent_word_count
            else:
                current_chunk.append(sent)
                current_length += sent_word_count
        
        # Add the last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk).strip()
            if len(chunk_text) > 30:
                text_chunks.append(chunk_text)

    doc.close()
    return text_chunks