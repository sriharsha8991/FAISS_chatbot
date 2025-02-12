# app.py
import streamlit as st
import os
import tempfile
from typing import List
 
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain.schema import Document
from langchain.prompts import PromptTemplate
 
# Configure Streamlit page
st.set_page_config(
    page_title="Document Q&A Bot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
# Add custom CSS
st.markdown("""
    <style>
        .reportview-container {
            margin-top: -2em;
        }
        .stButton>button {
            width: 100%;
        }
        .stAlert {
            margin-top: 1em;
            margin-bottom: 1em;
        }
    </style>
""", unsafe_allow_html=True)
 
# Sidebar
with st.sidebar:
    st.title("📚 Document Q&A Bot")
    st.markdown("""
    ### Instructions:
    1. Upload your document (PDF, DOCX, or TXT)
    2. Wait for processing
    3. Ask questions about the document
   
    ### Features:
    - PDF, DOCX, and TXT support
    - Advanced text processing
    - Context-aware responses
    - Chat history
   
    ### About:
    This bot uses RAG (Retrieval Augmented Generation) to provide accurate answers based on your documents.
    """)
 
# Initialize session state variables
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'error' not in st.session_state:
    st.session_state.error = None
if 'current_doc' not in st.session_state:
    st.session_state.current_doc = None
 
def init_groq() -> ChatGroq:
    """Initialize the Groq LLM with appropriate parameters."""
    api_key = st.secrets["GROQ_API_KEY"]
    return ChatGroq(
        api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.5,
        max_tokens=1024,
        top_p=0.9,
        presence_penalty=0.1,
        frequency_penalty=0.1
    )
 
def init_embeddings() -> HuggingFaceEmbeddings:
    """Initialize the embeddings model."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
 
def get_system_prompt() -> str:
    """Return the system prompt template."""
    return """You are a helpful AI assistant that answers questions based on the provided documents.
   
    Guidelines for your responses:
    1. Only answer based on the information from the provided document context
    2. If the answer cannot be found in the documents, say "I cannot find this information in the provided documents"
    3. Keep responses concise and to the point
    4. If relevant, cite specific sections from the document
    5. Maintain a professional and helpful tone
    6. If asked about topics outside the document, remind the user you can only answer questions about the uploaded document
   
    Context from documents: {context}
   
    Current conversation history: {chat_history}
   
    Question: {question}
   
    Helpful Answer:"""
 
def process_document(file_path: str, file_type: str) -> List[Document]:
    """Process document based on file type."""
    try:
        if file_type.lower() == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_type.lower() == '.docx':
            loader = Docx2txtLoader(file_path)
        elif file_type.lower() == '.txt':
            loader = TextLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
       
        documents = loader.load()
        if not documents:
            raise ValueError("No text could be extracted from the document.")
        return documents
    except Exception as e:
        raise Exception(f"Error loading document: {str(e)}")
 
def process_file(uploaded_file):
    """Process the uploaded file and create vector store."""
    if uploaded_file.name == st.session_state.current_doc:
        return st.session_state.vector_store
   
    file_extension = os.path.splitext(uploaded_file.name)[1]
   
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
 
    try:
        # Load and process document
        documents = process_document(tmp_file_path, file_extension)
       
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
 
        if not chunks:
            raise ValueError("Document was processed but no usable text chunks were created.")
 
        # Create vector store
        embeddings = init_embeddings()
        vector_store = FAISS.from_documents(chunks, embeddings)
       
        st.session_state.current_doc = uploaded_file.name
        return vector_store
 
    except Exception as e:
        st.session_state.error = str(e)
        return None
    finally:
        try:
            os.unlink(tmp_file_path)
        except:
            pass
 
def get_conversation_chain(vector_store):
    """Create conversation chain with vector store."""
    llm = init_groq()
   
    # Create QA chain with custom prompt
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={
            'k': 3,
            'fetch_k': 5,
            'maximal_marginal_relevance': True,
        }),
        return_source_documents=True,
        verbose=False,
        combine_docs_chain_kwargs={'prompt': PromptTemplate(
            input_variables=['context', 'question', 'chat_history'],
            template=get_system_prompt()
        )}
    )
   
    return qa_chain
 
# Main UI
st.title("Document Q&A Bot")
 
# File upload
uploaded_file = st.file_uploader(
    "Upload your document",
    type=['pdf', 'docx', 'txt'],
    help="Upload a PDF, DOCX, or TXT file to begin"
)
 
# Process uploaded file
if uploaded_file:
    with st.spinner("Processing document... This may take a minute."):
        st.session_state.vector_store = process_file(uploaded_file)
       
    if st.session_state.vector_store:
        st.success("Document processed successfully! You can now ask questions.")
    elif st.session_state.error:
        st.error(st.session_state.error)
        st.session_state.error = None
 
# Chat interface
if st.session_state.vector_store is not None:
    col1, col2 = st.columns([4, 1])
   
    with col2:
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.experimental_rerun()
   
    with col1:
        user_question = st.text_input(
            "Ask a question about your document:",
            placeholder="Enter your question here...",
            key="question_input"
        )
   
    if user_question:
        chain = get_conversation_chain(st.session_state.vector_store)
       
        with st.spinner("Generating response..."):
            try:
                response = chain({
                    'question': user_question,
                    'chat_history': st.session_state.chat_history
                })
               
                st.session_state.chat_history.append((user_question, response['answer']))
 
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
 
    # Display chat history
    if st.session_state.chat_history:
        st.subheader("Chat History")
        for i, (question, answer) in enumerate(st.session_state.chat_history):
            st.markdown(f"**Q{i+1}: {question}**")
            st.markdown(f"A{i+1}: {answer}")
            st.divider()
 
# Initial instructions
else:
    st.info("👆 Please upload a document to start asking questions!")