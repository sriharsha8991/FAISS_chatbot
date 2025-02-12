import streamlit as st
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="Healthcare Assistant",
    page_icon="👨‍⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for attractive UI
st.markdown("""
    <style>
        /* Main container styling */
        .main {
            padding: 2rem;
        }
        
        /* Custom title styling */
        .title-container {
            background-color: #f0f7ff;
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            text-align: center;
        }
        
        /* Chat container styling */
        .chat-container {
            background-color: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            margin-bottom: 1rem;
        }
        
        /* User message styling */
        .user-message {
            background-color: #e3f2fd;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        }
        
        /* Bot message styling */
        .bot-message {
            background-color: #f5f5f5;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        }
        
        /* Category indicators */
        .category-general {
            background-color: #c8e6c9;
            padding: 0.5rem;
            border-radius: 5px;
            color: #2e7d32;
            display: inline-block;
            margin-bottom: 0.5rem;
        }
        
        .category-moderate {
            background-color: #fff3e0;
            padding: 0.5rem;
            border-radius: 5px;
            color: #ef6c00;
            display: inline-block;
            margin-bottom: 0.5rem;
        }
        
        .category-hard {
            background-color: #ffebee;
            padding: 0.5rem;
            border-radius: 5px;
            color: #c62828;
            display: inline-block;
            margin-bottom: 0.5rem;
        }
        
        /* Custom button styling */
        .stButton > button {
            background-color: #1976d2;
            color: white;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            border: none;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            background-color: #1565c0;
            transform: translateY(-2px);
        }
        
        /* Disclaimer styling */
        .disclaimer {
            background-color: #fff3e0;
            padding: 1rem;
            border-radius: 5px;
            margin-top: 2rem;
            font-size: 0.9rem;
            border-left: 4px solid #ff9800;
        }
        
        /* Custom header styling */
        h1, h2, h3 {
            color: #1976d2;
            font-family: 'Arial', sans-serif;
        }
    </style>
""", unsafe_allow_html=True)

def get_api_key():
    """Get API key from environment variables."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("No API key found. Please set GROQ_API_KEY in .env file.")
        st.stop()
    return api_key

def init_llm():
    """Initialize the Groq LLM."""
    api_key = get_api_key()
    return ChatGroq(
        api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=1024,
        top_p=0.9
    )

def get_medical_prompt():
    """Return the medical conversation prompt template."""
    return """You are an AI medical assistant. Analyze the query and respond in ONE of these THREE formats ONLY, based on severity:

    1. For GENERAL (Simple) queries - Mild issues like common cold, mild headache:
    🟢 General (Simple)
    [Provide a quick, 1-2 sentence solution with basic home remedies or OTC medicines]

    2. For MODERATE (Intermediate) queries - Ongoing but non-emergency issues:
    🟡 Moderate (Intermediate)
    [Give a brief 2-3 sentence explanation of possible causes + simple medical advice + when to see doctor]

    3. For HARD (Serious) queries - Emergency situations:
    🔴 Hard (Serious)
    [Provide an urgent 1-2 sentence warning + specific immediate action to take]

    IMPORTANT RULES:
    - Choose ONLY ONE category
    - Keep responses brief and focused
    - For serious issues, always emphasize immediate medical attention
    - Include a short medical disclaimer for serious conditions

    User Query: {user_input}

    Response:"""

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Sidebar content
with st.sidebar:
    st.markdown("""
        <h2 style='text-align: center;'>👨‍⚕️ Healthcare Assistant</h2>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### About
    I'm your AI healthcare assistant, here to provide:
    - Quick medical guidance
    - Symptom assessment
    - Health recommendations
    
    ### Response Categories
    🟢 **General (Simple)**
    - Quick solutions for mild issues
    
    🟡 **Moderate (Intermediate)**
    - Brief explanation + basic medical advice
    
    🔴 **Hard (Serious)**
    - Urgent warning + immediate action
    
    ### How to Use
    1. Type your health concern
    2. Get instant medical guidance
    3. Follow recommended actions
    """)
    
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# Main content area
st.markdown("""
    <div class='title-container'>
        <h1>👨‍⚕️ Your Personal Healthcare Assistant</h1>
        <p>Get quick, focused medical guidance for your health concerns</p>
    </div>
""", unsafe_allow_html=True)

# Chat interface
user_input = st.text_input(
    "Describe your symptoms or health concern:",
    placeholder="Example: I've been having a mild headache since morning...",
    key="user_input"
)

if user_input:
    try:
        # Initialize LLM and create chain
        llm = init_llm()
        prompt = PromptTemplate(
            input_variables=["user_input"],
            template=get_medical_prompt()
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        
        # Get response
        with st.spinner("Analyzing your concern..."):
            response = chain.run(user_input=user_input)
            st.session_state.chat_history.append({"user": user_input, "bot": response})
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Display chat history
if st.session_state.chat_history:
    for chat in reversed(st.session_state.chat_history):
        # User message
        st.markdown(f"""
            <div class='user-message'>
                <strong>You:</strong><br>{chat['user']}
            </div>
        """, unsafe_allow_html=True)
        
        # Bot message
        st.markdown(f"""
            <div class='bot-message'>
                <strong>Healthcare Assistant:</strong><br>{chat['bot']}
            </div>
        """, unsafe_allow_html=True)
else:
    # Welcome message
    st.markdown("""
        <div class='chat-container'>
            <h3>👋 Welcome to Your Healthcare Assistant!</h3>
            <p>I provide quick, focused guidance for your health concerns:</p>
            <ul>
                <li>Simple remedies for mild symptoms</li>
                <li>Brief medical advice for ongoing issues</li>
                <li>Urgent guidance for serious conditions</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

# Disclaimer
st.markdown("""
    <div class='disclaimer'>
        <strong>⚠️ Important Disclaimer:</strong><br>
        This AI healthcare assistant provides general medical information and guidance only. 
        It is not a substitute for professional medical advice, diagnosis, or treatment. 
        Always seek the advice of your physician or other qualified health provider with any 
        questions you may have regarding a medical condition.
    </div>
""", unsafe_allow_html=True)