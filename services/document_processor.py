from io import BytesIO

from pypdf import PdfReader


def extract_text_from_pdf(pdf_file):
    """
    Extract text from an uploaded PDF file.
    """

    pdf_bytes = pdf_file.getvalue()

    pdf_reader = PdfReader(BytesIO(pdf_bytes))

    extracted_pages = []

    for page_number, page in enumerate(pdf_reader.pages, start=1):
        page_text = page.extract_text() or ""

        if page_text.strip():
            extracted_pages.append(
                {
                    "page_number": page_number,
                    "text": page_text.strip(),
                }
            )

    return extracted_pages


def split_text_into_chunks(text, chunk_size=1000, overlap=200):
    """
    Split text into smaller, overlapping chunks.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    words = text.split()

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))

        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        if end == len(words):
            break

        start = end - overlap

    return chunks