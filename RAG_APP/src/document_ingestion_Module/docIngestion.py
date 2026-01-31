from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from typing import List
import tempfile
import os
from pathlib import Path



class DocumentIngestion:

    """ Handles the PDF loading and text chunking"""



    def __init__(self , chunk_size: int, chunk_overlap: int ):

        """ Initialize the doc ingestion module
        
        Args: chunk_size : size of each text chunk
                chunk_overlap : overlap between text chunks
        
        
        
        """

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap = self.chunk_overlap,
            length_function = len

        )


    def extract_text_from_pdf(self, pdf_path: str) :



        """ Extracts text chunks from a PDF file
        
        Args:
            pdf_file_path (str): Path to the PDF file. """
        

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        chunks  = self.text_splitter.split_documents(documents)
        return chunks
    

    
        

        





    






















        