# Document Q&A Bot

An intelligent document question-answering system built with Streamlit and LangChain that enables users to have interactive conversations about their documents.

## Features

- **Multi-Format Support**: Process PDF, DOCX, and TXT files
- **Interactive Chat Interface**: Natural conversation flow with document context
- **RAG Implementation**: Uses Retrieval Augmented Generation for accurate answers
- **Advanced Processing**: Chunks documents efficiently for better context retrieval
- **Responsive UI**: Clean and intuitive Streamlit interface

## Tech Stack

- **Frontend**: Streamlit
- **LLM**: Groq (llama-3.3-70b-versatile)
- **Embeddings**: HuggingFace (sentence-transformers/all-MiniLM-L6-v2)
- **Vector Store**: FAISS
- **Document Processing**: LangChain

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd medical-bot-kaustub
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.streamlit/secrets.toml` file with:
```toml
GROQ_API_KEY = "your-groq-api-key"
```

4. Run the application:
```bash
streamlit run app.py
```



## Usage

1. Launch the application
2. Upload a document (PDF, DOCX, or TXT)
3. Wait for processing completion
4. Start asking questions about the document
5. View chat history and responses

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License

## Acknowledgments

- LangChain for the document processing pipeline
- Groq for LLM capabilities
- Streamlit for the web interface
