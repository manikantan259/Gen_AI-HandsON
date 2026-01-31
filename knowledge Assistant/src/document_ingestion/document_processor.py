from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


from typing import List , Union
from pathlib import Path
from langchain_community.document_loaders import(
    WebBaseLoader ,
    PyPDFLoader ,
    TextLoader ,
    PyPDFDirectoryLoader
)


""" upload documents """

import tempfile
import os
from typing import BinaryIO

"""  Save DOc """
import os
import uuid


from pathlib import Path

DEFAULT_DATA_DIR = Path(
    r"C:\Users\rajku\OneDrive\Desktop\Assignement\Week_2\knowledge Assistant\data"
)



class DocumentProcessor:
    """A class to process and split documents from various sources."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def load_document(self, source: Union[str, Path], source_type: str) -> List[Document]:
        """Load documents from various sources."""
        if source_type == "web":
            loader = WebBaseLoader(str(source))
        elif source_type == "pdf":
            loader = PyPDFLoader(str(source))
        elif source_type == "text":
            loader = TextLoader(str(source))
        elif source_type == "pdf_directory":
            loader = PyPDFDirectoryLoader(str(source))
        else:
            raise ValueError(f"Unsupported source type: {source_type}")

        documents = loader.load()
        return documents
    

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks."""
        split_docs = []
        for doc in documents:
            chunks = self.text_splitter.split_text(doc.page_content)
            for chunk in chunks:
                split_docs.append(Document(page_content=chunk, metadata=doc.metadata))
        return split_docs
    
    def process(self, source: Union[str, Path], source_type: str) -> List[Document]:
        """Load and split documents from the given source."""
        documents = self.load_document(source, source_type)
        split_docs = self.split_documents(documents)
        return split_docs
    
    def process_uploaded_file(self, file: BinaryIO, file_type: str) -> List[Document]:
        """Process an uploaded file and store it in the specified path."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file.read())
            temp_file_path = temp_file.name
        try:
            documents = self.load_document(temp_file_path, file_type)
            split_docs = self.split_documents(documents)
            # Store the processed documents in the specified path
            self.save_and_process_uploaded_file(file.read(), file.name)
        finally:
            os.remove(temp_file_path)

        return split_docs
    
    def save_and_process_uploaded_file(self,file_bytes: bytes,filename: str,data_dir: Path = DEFAULT_DATA_DIR) -> List[Document]:
    

        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        suffix = Path(filename).suffix.lower()

    # prevent overwrite → unique filename
        unique_name = f"{Path(filename).stem}_{uuid.uuid4().hex}{suffix}"
        file_path = data_dir / unique_name

    # save file
        with open(file_path, "wb") as f:
            f.write(file_bytes)

    # process based on file type
        if suffix == ".pdf":
         documents = self.load_document(file_path, "pdf")
        elif suffix == ".txt":
            documents = self.load_document(file_path, "text")
        else:
            raise ValueError("Unsupported uploaded file type")
        return self.split_documents(documents)
