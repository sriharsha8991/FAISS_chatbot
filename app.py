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
    page_title="MediBot - Medical Document Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for medical theme
st.markdown("""
    <style>
        /* Medical color scheme */
        :root {
            --medical-blue: #0077cc;
            --light-blue: #e6f3ff;
            --medical-red: #ff4444;
        }
        
        /* Header styling */
        .main .block-container {
            padding-top: 2rem;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: var(--light-blue);
        }
        
        /* Button styling */
        .stButton>button {
            width: 100%;
            background-color: var(--medical-blue) !important;
            color: white !important;
            border-radius: 25px;
        }
        
        /* Chat container styling */
        .chat-container {
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 1rem;
            margin: 1rem 0;
            background-color: white;
        }
        
        /* Question styling */
        .question {
            background-color: var(--light-blue);
            padding: 1rem;
            border-radius: 15px;
            margin: 0.5rem 0;
        }
        
        /* Answer styling */
        .answer {
            background-color: white;
            padding: 1rem;
            border-radius: 15px;
            margin: 0.5rem 0;
            border-left: 4px solid var(--medical-blue);
        }
        
        /* Alert styling */
        .stAlert {
            border-radius: 10px;
            margin: 1rem 0;
        }
        
        /* Input field styling */
        .stTextInput>div>div>input {
            border-radius: 25px;
        }
        
        /* Custom title styling */
        .medical-title {
            color: var(--medical-blue);
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 1rem;
        }
        
        /* Custom icon styling */
        .icon-container {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<div class="icon-container"><h1>🏥 MediBot</h1></div>', unsafe_allow_html=True)
    st.markdown("""
    ### How it Works:
    1. 📄 Upload your medical document
    2. ⏳ Wait for processing
    3. 💬 Ask medical questions
    
    ### Features:
    - 📊 Medical document analysis
    - 🔍 Intelligent context understanding
    - 📝 Detailed medical insights
    - 📚 Secure document handling
    
    ### Supported Formats:
    - Medical Reports (PDF)
    - Clinical Notes (DOCX)
    - Lab Results (TXT)
    
    ### Important Note:
    This bot is for informational purposes only and should not replace professional medical advice.
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
