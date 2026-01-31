"""Streamlit UI for Agentic RAG System - Simplified Version"""

import streamlit as st
from pathlib import Path
import sys
import time

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graphbuilder import GraphBuilder

# Page configuration
st.set_page_config(
    page_title="🤖 RAG Search",
    page_icon="🔍",
    layout="centered"
)

# Simple CSS
st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables"""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'history' not in st.session_state:
        st.session_state.history = []

@st.cache_resource
def initialize_rag():
    """Initialize the RAG system (cached)"""
    try:
        # Initialize components
        config = Config()
        llm = config.get_llm()
        doc_processor = DocumentProcessor(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        # Use a fresh vector store with a unique name to avoid caching issues
        import uuid
        vector_store = VectorStore(persist_directory=f"vectorstore_data_{uuid.uuid4().hex[:8]}")
        
        all_documents = []
        
        # Load documents from data folder (PDFs)
        data_dir = Path("data")
        if data_dir.exists():
            pdf_files = list(data_dir.glob("*.pdf"))
            st.write(f"Found {len(pdf_files)} PDF files in data folder")
            for pdf_file in pdf_files:
                try:
                    st.write(f"Loading {pdf_file.name}...")
                    documents = doc_processor.process(str(pdf_file), "pdf")
                    st.write(f"  - Loaded {len(documents)} chunks from {pdf_file.name}")
                    all_documents.extend(documents)
                except Exception as e:
                    st.warning(f"Error loading {pdf_file}: {e}")
        
        # Load documents from URLs
        urls = config.DEFAULT_URLS
        for url in urls:
            try:
                st.write(f"Loading from {url}...")
                documents = doc_processor.process(url, "web")
                st.write(f"  - Loaded {len(documents)} chunks")
                all_documents.extend(documents)
            except Exception as e:
                st.warning(f"Error loading {url}: {e}")
        
        # Add documents to vector store
        if all_documents:
            st.write(f"Adding {len(all_documents)} total chunks to vector store...")
            vector_store.add_documents(all_documents)
            st.write("✅ Vector store created successfully!")
        else:
            st.warning("No documents were loaded!")
        
        # Build graph
        graph_builder = GraphBuilder(
            retriever=vector_store.get_retriever(),
            llm=llm
        )
        
        return graph_builder, len(all_documents)
    except Exception as e:
        st.error(f"Failed to initialize: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None, 0
        return None, 0

def main():
    """Main application"""
    init_session_state()
    
    # Title
    st.title("🔍 RAG Document Search")
    st.markdown("Ask questions about the loaded documents")
    
    # Initialize system
    if not st.session_state.initialized:
        with st.spinner("Loading system..."):
            rag_system, num_chunks = initialize_rag()
            if rag_system:
                st.session_state.rag_system = rag_system
                st.session_state.initialized = True
                st.success(f"✅ System ready! ({num_chunks} document chunks loaded)")
    
    st.markdown("---")
    
    # File uploader for document upload
    st.markdown("### 📤 Upload Documents")
    uploaded_file = st.file_uploader("Upload a document", type=['pdf', 'docx', 'txt'])
    if uploaded_file is not None:
        # Process the uploaded file
        st.write("File uploaded successfully!")
        # You can add your file processing logic here
        # For example, read the file and display its content
        if st.session_state.rag_system:
            with st.spinner("Processing document..."):
                try:
                    doc_processor = DocumentProcessor()
                    # Get file type from extension
                    file_extension = uploaded_file.name.split('.')[-1].lower()
                    file_type_map = {'pdf': 'pdf', 'txt': 'text', 'docx': 'docx'}
                    file_type = file_type_map.get(file_extension, 'text')
                    
                    # Process uploaded file
                    documents = doc_processor.save_and_process_uploaded_file(
                        uploaded_file.read(),
                        uploaded_file.name
                    )
                    st.success(f"✅ Document processed! ({len(documents)} chunks created)")
                    
                    # Clear the RAG cache to reload documents including the new upload
                    st.cache_resource.clear()
                    st.info("Cache cleared. Please refresh the page to load the new document into the vector store.")
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
    
    st.markdown("---")
    
    # Search interface
    with st.form("search_form"):
        question = st.text_input(
            "Enter your question:",
            placeholder="What would you like to know?"
        )
        submit = st.form_submit_button("🔍 Search")
    
    # Process search
    if submit and question:
        if st.session_state.rag_system:
            with st.spinner("Searching..."):
                start_time = time.time()
                
                # Get answer
                result = st.session_state.rag_system.run(question)
                
                elapsed_time = time.time() - start_time
                
                # Extract answer and docs from result
                answer = result.get('answer', '')
                retrieved_docs = result.get('retrieved_documents', [])
                
                # Add to history
                st.session_state.history.append({
                    'question': question,
                    'answer': answer,
                    'time': elapsed_time
                })
                
                # Display answer
                st.markdown("### 💡 Answer")
                st.success(answer)
                
                # Show retrieved docs in expander
                with st.expander("📄 Source Documents"):
                    for i, doc in enumerate(retrieved_docs, 1):
                        st.text_area(
                            f"Document {i}",
                            doc.page_content[:300] + "...",
                            height=100,
                            disabled=True
                        )
                
                st.caption(f"⏱️ Response time: {elapsed_time:.2f} seconds")
    
    # Show history
    if st.session_state.history:
        st.markdown("---")
        st.markdown("### 📜 Recent Searches")
        
        for item in reversed(st.session_state.history[-3:]):  # Show last 3
            with st.container():
                st.markdown(f"**Q:** {item['question']}")
                st.markdown(f"**A:** {item['answer'][:200]}...")
                st.caption(f"Time: {item['time']:.2f}s")
                st.markdown("")

if __name__ == "__main__":
    main()