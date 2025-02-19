# 🤖 LangChain RAG with DeepSeek & Pinecone

A powerful Retrieval-Augmented Generation (RAG) application built with LangChain, DeepSeek (Ollama), and Pinecone. This project demonstrates how to create an efficient document question-answering system with state-of-the-art components.

![Python Version](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen)

## 🌟 Features

- **Advanced RAG Implementation**: Utilizes LangChain's latest RAG patterns for optimal retrieval and generation
- **DeepSeek Integration**: Leverages DeepSeek's powerful language model through Ollama for high-quality responses
- **Pinecone Vector Store**: Efficient similarity search and document retrieval using Pinecone's vector database
- **Interactive UI**: Clean and intuitive interface built with Streamlit
- **Document Processing**: Supports processig of folders of documents in HTML format
- **Context-Aware Responses**: Generates accurate answers based on retrieved document context
- **Full Tracing**: Trace all the LLM calls using LangSmith
- **Easy Settingup**: Use [uv](https://docs.astral.sh/uv/) to  set up the project environment

## 🚀 Getting Started

### Prerequisites

- Python 3.12
- Ollama installed locally
- Pinecone API key (and account)
- Required Python packages:
```bash
uv sync
```

### Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/AlfredoCinelli/langchain-assistant
cd langchain-assistant
```

2. Create a `.env` file with your credentials in the `local` directory:
```bash
PINECONE_API_KEY=your-api-key-here
INDEX_NAME="index-name"
HUGGINGFACE_API_KEY="your-hf-api-key"
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="your-langchain-api-key"
LANGCHAIN_PROJECT="project-name"
```

3. Pull the DeepSeek model using Ollama:
```bash
ollama pull deepseek-r1[:1.5b/7b/8b/14b]
```

### Running the Application

See the below usage guide to fully setup the project.

1. Start the Streamlit application (after setting up the entire project):
```bash
make app.py
```

2. Navigate to `http://localhost:8501` in your browser

## 🛠️ Technical Architecture

```
repo/
├── app.py          # Streamlit application entry src
├── src/
│   ├── scraping    # Package to scrape Langchain docs (async and optimised)
│   ├── ingestion   # Package to ingest scraped HTML, process them and ingest in Pinecone
│   └── backend     # Package with all the core functions and classes
│   ├── tools.py    # Module with tools
├── Makefile        # Makefile to run modules with quick syntax and uv robustness
├── pyproject.toml  # Python project configuration
```

## 💡 Usage

1. **Scrape Documents**: Use the modules defined in the `src/scraping` folder to scrape Langchain docs (`make scraper_docs` & `make scraper_api`)
2. **Index Creation**: Documents are automatically processed and indexed in Pinecone via the `src/ingestion/ingestion.py` module (`make ingestion`)
3. **Run App**: Spin the Streamlit server app (`make app.py`)
4. **Ask Questions**: Ask questions to the LLM, see the reasoning trace of `DeepSeek`, the answer and live `sources`

## ⚙️ Configuration

Customize the application behavior in `config.py`:

- Chunk size and overlap for document processing
- Number of relevant chunks to retrieve
- LLM temperature and other parameters
- Vector store configuration


## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://python.langchain.com/docs/get_started/introduction) for the excellent framework
- [DeepSeek](https://github.com/deepseek-ai/DeepSeek-LLM) for the powerful language model
- [Pinecone](https://www.pinecone.io/) for vector similarity search
- [Streamlit](https://streamlit.io/) for the user interface framework
- [Ollama](https://ollama.ai/) for local model deployment
- [LangSmith](https://smith.langchain.com/) for tracing and monitoring
- [uv](https://docs.astral.sh/uv/) to  set up the project environment

## 📧 Contact

For questions and feedback, please open an issue or contact [alfredocinelli96@gmail.com](mailto:alfredocinelli96@gmail.com).