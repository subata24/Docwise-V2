import os, uuid
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")


def ingest_document(file_path: str, doc_id: str, user_id: str) -> dict:
    extension = Path(file_path).suffix.lower()

    if extension == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader

        loader = PyPDFLoader(file_path)
        raw_docs = loader.load()

    elif extension == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader

        loader = Docx2txtLoader(file_path)
        raw_docs = loader.load()

    elif extension in {".txt", ".md"}:
        from langchain_community.document_loaders import TextLoader

        loader = TextLoader(file_path, encoding="utf-8")
        raw_docs = loader.load()

    elif extension == ".png":
        from PIL import Image
        import pytesseract

        with Image.open(file_path) as image:
            text = pytesseract.image_to_string(image)

        if not text.strip():
            raise ValueError("OCR could not extract readable text from the image")

        raw_docs = [
            Document(
                page_content=text,
                metadata={}
            )
        ]

    else:
        raise ValueError(f"Unsupported file type: {extension}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = splitter.split_documents(raw_docs)

    for chunk in chunks:
        chunk.metadata.update({
            "doc_id": doc_id,
            "user_id": user_id,
            "source": Path(file_path).name,
            "file_type": extension.lstrip(".")
        })

    vectorstore = Chroma(
        collection_name=f"user_{user_id}",
        embedding_function=EMBED_MODEL,
        persist_directory=CHROMA_DIR
    )

    vectorstore.add_documents(chunks)

    return {
        "doc_id": doc_id,
        "chunks_added": len(chunks),
        "pages": len(raw_docs) if extension == ".pdf" else None
    }


def get_vectorstore(user_id: str) -> Chroma:
    return Chroma(
        collection_name=f"user_{user_id}",
        embedding_function=EMBED_MODEL,
        persist_directory=CHROMA_DIR
    )