# app.py
import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Page configuration
st.set_page_config(
    page_title="MediChat",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced CSS for chat interface (same as before)
st.markdown("""
    <style>
        /* Main container */
        .main > div {
            padding-top: 0;
        }
        
        /* Chat container */
        .chat-container {
            max-width: 800px;
            margin: 0 auto;
            padding: 1rem;
            height: calc(100vh - 100px);
            display: flex;
            flex-direction: column;
        }
        
        /* Header */
        .chat-header {
            position: sticky;
            top: 0;
            background: var(--background-color);
            padding: 1rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            z-index: 100;
            backdrop-filter: blur(10px);
        }
        
        /* Messages area */
        .chat-messages {
            flex-grow: 1;
            overflow-y: auto;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 10px;
            background: rgba(0, 0, 0, 0.2);
        }
        
        /* Message bubbles */
        .user-message {
            background: #2C3E50;
            color: white;
            border-radius: 20px 20px 5px 20px;
            padding: 1rem;
            margin: 1rem 0;
            max-width: 80%;
            margin-left: auto;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        .bot-message {
            background: #34495E;
            color: white;
            border-radius: 20px 20px 20px 5px;
            padding: 1rem;
            margin: 1rem 0;
            max-width: 80%;
            margin-right: auto;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        /* Severity indicators */
        .severity-high {
            border-left: 4px solid #FF4444;
        }
        
        .severity-medium {
            border-left: 4px solid #FFA500;
        }
        
        .severity-low {
            border-left: 4px solid #28a745;
        }
        
        /* Input area */
        .chat-input {
            position: sticky;
            bottom: 0;
            background: var(--background-color);
            padding: 1rem;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
        }
        
        /* Custom styles */
        .stTextInput > div > div > input {
            border-radius: 20px !important;
            padding: 10px 20px !important;
            background: rgba(255, 255, 255, 0.1) !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            color: white !important;
        }
        
        .stButton > button {
            border-radius: 20px !important;
            padding: 10px 20px !important;
            background: #0077cc !important;
            color: white !important;
            border: none !important;
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Emergency badge */
        .emergency-badge {
            background: #FF4444;
            color: white;
            padding: 0.2rem 0.5rem;
            border-radius: 10px;
            font-size: 0.8rem;
            margin-left: 0.5rem;
        }
        
        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 3px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.2);
            border-radius: 3px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.3);
        }
    </style>
""", unsafe_allow_html=True)
with st.sidebar:
    st.title("🏥 MediChat")
    st.markdown("""
    ### Instructions:
    1. Describe your medical concern
    2. Wait for the assistant's response
    3. Follow the assistant's guidance
    
    ### Features:
    - Medical pre-screening
    - Severity assessment
    - Chat history with memory
    
    ### About:
    This AI assistant helps assess the severity of your medical concerns based on your symptoms.
    """)
# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

def init_llm():
    """Initialize the Groq LLM."""
    return ChatGroq(
        api_key="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",
        model_name="llama-3.3-70b-versatile",
        temperature=0.7
    )
def get_conversation_chain():
    """Create a conversation chain with memory."""
    llm = init_llm()
    
    return ConversationChain(
        llm=llm,
        memory=st.session_state.memory,
        prompt=PromptTemplate(
            input_variables=["chat_history", "input"],
            template=MEDICAL_TEMPLATE
        ),
        verbose=False
    )
# Enhanced prompt template with medical context and chat history
MEDICAL_TEMPLATE = """You are a concise and focused medical pre-screening assistant. Provide brief, clear responses while considering the conversation history.

Guidelines:
1. Keep responses short and direct (2-3 sentences for normal cases)
2. Only mention severity if it's HIGH
3. Ask at most one follow-up question if needed
4. Don't repeat previous information unless there's a significant change
5. Focus on new information in each response

Previous conversation:
{chat_history}

Current symptoms: {input}

Additional guidelines:
- Don't start every response with "I see" or "I understand"
- Don't repeat the symptoms just mentioned
- Don't say "I'm glad you're reaching out" in every response
- Only mention "seek immediate medical attention" for HIGH severity cases
- Keep responses conversational but professional

Your response should be brief and focused on either:
1. Addressing the new symptom
2. Asking one specific follow-up question
3. Providing clear guidance if severity is HIGH

Response:"""

def determine_severity(symptoms: list, response: str) -> str:
    """Determine severity level from symptoms and response."""
    # List of high-severity indicators
    high_severity_symptoms = [
        "chest pain", "difficulty breathing", "severe pain",
        "unconscious", "seizure", "stroke", "heart attack"
    ]
    
    # List of medium-severity indicators
    medium_severity_symptoms = [
        "fever", "persistent", "worsening", "moderate pain",
        "infection", "vomiting", "diarrhea"
    ]
    
    response_upper = response.upper()
    
    # Check for high severity conditions
    if any(symptom in " ".join(symptoms).lower() for symptom in high_severity_symptoms):
        return "HIGH"
    elif "HIGH" in response_upper or "EMERGENCY" in response_upper or "IMMEDIATE" in response_upper:
        return "HIGH"
    # Check for medium severity conditions
    elif any(symptom in " ".join(symptoms).lower() for symptom in medium_severity_symptoms):
        return "MEDIUM"
    elif "MEDIUM" in response_upper or "MODERATE" in response_upper:
        return "MEDIUM"
    return "LOW"

def process_user_input():
    """Process user input and generate response with memory."""
    if st.session_state.user_input:
        user_message = st.session_state.user_input.strip()
        
        # Update symptoms list
        if 'symptoms' not in st.session_state:
            st.session_state.symptoms = []
        st.session_state.symptoms.append(user_message)
        
        # Add user message to display
        st.session_state.messages.append({
            "role": "user",
            "content": user_message
        })
        
        # Clear input
        st.session_state.user_input = ""
        
        try:
            # Get response using conversation chain with memory
            conversation = get_conversation_chain()
            response = conversation.predict(input=user_message)
            
            # Determine severity based on all symptoms and response
            severity = determine_severity(st.session_state.symptoms, response)
            
            # Add assistant response to display
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "severity": severity
            })
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

def clear_chat():
    """Clear chat history and memory."""
    st.session_state.messages = []
    st.session_state.symptoms = []
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
# UI Components

# Chat header
st.markdown("""
    <div class="chat-header">
        <h1>🏥 MediChat Assistant</h1>
        <p>Your AI Medical Pre-screening Assistant with Memory</p>
    </div>
""", unsafe_allow_html=True)
with st.container():
# Chat messages container
    st.markdown('<div class="chat-messages">', unsafe_allow_html=True)

    # Display welcome message if no messages
    if not st.session_state.messages:
        st.markdown("""
            <div class="bot-message">
                👋 Hello! I'm your medical pre-screening assistant. I can remember our conversation to provide better guidance.
                Please describe your medical concern, and I'll help assess its severity.
                
                ⚠️ Remember: For medical emergencies, call emergency services immediately!
            </div>
        """, unsafe_allow_html=True)

    # Display chat history with context awareness
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
                <div class="user-message">
                    {message["content"]}
                </div>
            """, unsafe_allow_html=True)
        else:
            severity_class = f"severity-{message.get('severity', 'low').lower()}"
            severity_badge = ""
            if message.get('severity') == "HIGH":
                severity_badge = '<span class="emergency-badge">URGENT</span>'
            
            st.markdown(f"""
                <div class="bot-message {severity_class}">
                    {severity_badge}
                    {message["content"]}
                </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Chat input and controls
st.markdown('<div class="chat-input">', unsafe_allow_html=True)
col1, col2 = st.columns([4, 1])

with col1:
    st.text_input(
        "",
        placeholder="Describe your medical concern...",
        key="user_input",
        on_change=process_user_input
    )

with col2:
    if st.session_state.messages and st.button("Clear Chat"):
        clear_chat()

st.markdown('</div>', unsafe_allow_html=True)

# Medical disclaimer
st.markdown("""
    <div style="font-size: 0.8rem; color: rgba(255, 255, 255, 0.6); text-align: center; padding: 1rem;">
        ⚕️ Medical Disclaimer: This is an AI assistant for informational purposes only. 
        Always seek professional medical advice for health concerns.
        
        🔒 Your conversation history is used to provide better guidance but is not stored permanently.
    </div>
""", unsafe_allow_html=True)
