"""Streamlit UI for RAG System"""

import sys
from pathlib import Path
import time

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

import streamlit as st

try:
    from config_Module import config
    from document_ingestion_Module import docIngestion
    from embeddings_Module import embeddings
    from vectorStore_Module import vectorStore
    from Retrival_Module import retrieval
    from llm_call import llm
    
    GROQ_API_KEY = config.GROQ_API_KEY
    PINECONE_API_KEY = config.PINECONE_API_KEY
    EMBEDDING_MODEL = config.EMBEDDING_MODEL
    LLM_MODEL = config.LLM_MODEL
    CHUNK_SIZE = config.CHUNK_SIZE
    CHUNK_OVERLAP = config.CHUNK_OVERLAP
    DEFAULT_DATA_DIR = config.DEFAULT_DATA_DIR
    DocumentIngestion = docIngestion.DocumentIngestion
    EmbeddingModel = embeddings.Embeddings
    VectorDatabase = vectorStore.VectorDatabase
    Retrieval = retrieval.Retrieval
    LLMModel = llm.LLMMOdel

except ModuleNotFoundError as e:
    st.error(f"Module import error: {e}")
    st.info("Please ensure all modules are in the src/ directory with proper __init__.py files")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="🤖 RAG Search",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .answer-box {
        background-color: #e8f5e9;
        color: #1b5e20;
        padding: 20px;
        border-radius: 5px;
        border-left: 5px solid #4CAF50;
        font-size: 16px;
        line-height: 1.6;
    }
    </style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables"""
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'history' not in st.session_state:
        st.session_state.history = []

@st.cache_resource
def load_vector_store():
    """Auto-connect to existing Pinecone index"""
    try:
        st.write("🔗 Connecting to Pinecone index...")
        embedder = EmbeddingModel(model_name=EMBEDDING_MODEL)
        vector_db = VectorDatabase(
            index_name="ragapp",
            embeddding_model=embedder
        )
        # Load existing index
        vector_store = vector_db.load_existing_index()
        st.write("✅ Connected to Pinecone!")
        return vector_store, embedder
    except Exception as e:
        st.warning(f"⚠️ Could not connect to Pinecone: {str(e)}")
        return None, None

def main():
    """Main application"""
    init_session_state()
    
    # Title
    st.title("🔍 RAG Document Search System")
    st.markdown("Ask questions about your documents powered by Groq LLM")
    
    # Load vector store on startup
    if st.session_state.vector_store is None:
        with st.spinner("⏳ Connecting to vector store..."):
            vector_store, embedder = load_vector_store()
            if vector_store is not None:
                st.session_state.vector_store = vector_store
                st.session_state.initialized = True
    
    # Sidebar for document management
    with st.sidebar:
        st.header("📄 Document Management")
        
        pdf_files = list(DEFAULT_DATA_DIR.glob("*.pdf"))
        
        if pdf_files:
            st.markdown("### Available PDFs")
            selected_pdf = st.selectbox(
                "Select PDF to process:",
                options=[pdf.name for pdf in pdf_files]
            )
            
            if st.button("📤 Process New PDF", use_container_width=True):
                with st.spinner("🔄 Processing PDF..."):
                    try:
                        ingestion = DocumentIngestion(
                            chunk_size=CHUNK_SIZE,
                            chunk_overlap=CHUNK_OVERLAP
                        )
                        pdf_path = DEFAULT_DATA_DIR / selected_pdf
                        documents = ingestion.extract_text_from_pdf(str(pdf_path))
                        
                        embedder = EmbeddingModel(model_name=EMBEDDING_MODEL)
                        vector_db = VectorDatabase(
                            index_name="ragapp",
                            embeddding_model=embedder
                        )
                        st.session_state.vector_store = vector_db.add_documents(documents)
                        st.cache_resource.clear()
                        
                        st.sidebar.success(f"✅ Processed {len(documents)} chunks!")
                        st.session_state.initialized = True
                        st.rerun()
                        
                    except Exception as e:
                        st.sidebar.error(f"❌ Error: {str(e)}")
        else:
            st.warning("No PDFs found in data folder")
    
    st.divider()
    
    # Main search interface
    col1, col2 = st.columns([4, 1])
    
    with col1:
        question = st.text_input(
            "🤔 Ask a question:",
            placeholder="What would you like to know about the documents?"
        )
    with col2:
        search_button = st.button("🔍 Search", use_container_width=True)
    
    # Process search
    if (search_button or question) and question:
        if st.session_state.vector_store is None:
            st.error("❌ No vector store loaded. Please process a PDF first.")
        else:
            with st.spinner("🔎 Searching and generating answer..."):
                try:
                    start_time = time.time()
                    
                    # Retrieve documents
                    retriever = Retrieval(
                        vector_store=st.session_state.vector_store,
                        top_k=3
                    )
                    relevant_docs = retriever.search(question)
                    
                    # Generate answer
                    llm_model = LLMModel(api_key=GROQ_API_KEY, model_name=LLM_MODEL)
                    answer = llm_model.generate_answer(question, relevant_docs)
                    
                    elapsed_time = time.time() - start_time
                    
                    # Add to history
                    st.session_state.history.append({
                        'question': question,
                        'answer': answer,
                        'time': elapsed_time
                    })
                    
                    # Display answer
                    st.markdown("### 💡 Answer")
                    st.markdown(f"""
                    <div class="answer-box">
                    {answer}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show source documents
                    with st.expander("📚 Source Documents"):
                        for i, doc in enumerate(relevant_docs, 1):
                            st.markdown(f"**Document {i}:**")
                            st.text_area(
                                f"Source {i}",
                                doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                                height=120,
                                disabled=True,
                                label_visibility="collapsed"
                            )
                            st.divider()
                    
                    st.caption(f"⏱️ Response time: {elapsed_time:.2f} seconds")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    # Show search history
    if st.session_state.history:
        st.divider()
        st.markdown("### 📜 Recent Searches")
        
        for i, item in enumerate(reversed(st.session_state.history[-5:]), 1):
            with st.container():
                st.markdown(f"**Q {i}:** {item['question']}")
                st.markdown(f"**A {i}:** {item['answer'][:150]}...")
                st.caption(f"⏱️ {item['time']:.2f}s")
                st.markdown("")

if __name__ == "__main__":
    main()
