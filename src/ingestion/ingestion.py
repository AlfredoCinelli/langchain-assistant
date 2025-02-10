"""Module aimed at ingesting the scraped LangChain docs into the Pinecone vector store."""

# Import packages and modules

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, UnstructuredHTMLLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import os
import warnings
import logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "thenlper/gte-base" # bi-encoder model for embeddings generation

load_dotenv("local/.env")

PATHS = [
    {
        "name": "langchain-api", # scraped LangChain API docs (local folder path)
        "link_specs": {
            "prefix": {"old": "langchain-api/", "new": "https://python.langchain.com/api_reference/"},
            "suffix": None,
        },
    },
    {
        "name": "langchain-docs", # scraped LangChain docs (e.g., 'how to', 'concepts', 'tutorials')
        "link_specs": {
            "prefix": {"old": "langchain-docs/", "new": "https://python.langchain.com/"},
            "suffix": {"old": "index.html", "new": ""},
        }
    }
]

def source_to_link(
    source: str,
    prefix: dict[str, str] | None,
    suffix: dict[str, str] | None,
) -> str:
    """
    Convert a source string to a link.
    
    :param source: source of the file from local FS
    :type source: str
    :param prefix: prefix to add to the source
    :type prefix: dict[str, str] | None
    :param suffix: suffix to add to the source
    :type suffix: dict[str, str] | None
    :return: link to the source on the original web page
    :rtype: str
    """
    if prefix is not None:
        source = source.replace(prefix.get("old"), prefix.get("new"))
    if suffix is not None:
        source = source.replace(suffix.get("old"), suffix.get("new"))
    return source

def ingest_docs(
    path: str,
    link_specs: dict[str, dict[str, str] | None],
) -> None:
    """
    Main function to ingest scraped LangChain docs into the Pinecone vector store.
    
    :param path: path to the local folder where the scraped LangChain docs are stored
    :type path: str
    :param link_specs: specs to convert the source of the file from local FS to the link on the original web page
    :type link_specs: dict[str, dict[str, str] | None]
    """

    logger.info(f"Loading documents from {path}...")
    loader = DirectoryLoader(
        path=path,
        glob="**/*.html",
        loader_cls=UnstructuredHTMLLoader,
    )
    raw_docs = loader.load()
    logger.info(f"loaded {len(raw_docs)} documents")
    
    logger.info("Splitting documents via recursive character text splitter...")
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=1_000,
        chunk_overlap=50,
    )
    
    documents = splitter.split_documents(raw_docs)
    logger.info("Updating documents...")
    for doc in documents:
        source_url = source_to_link(
            source=doc.metadata["source"],
            **link_specs,
        )
        doc.metadata.update({"source": source_url})
        
    logger.info(f"Index {len(documents)} documents")
    
    logger.info("Initializing Pinecone...")
    pc = Pinecone(
        api_key=os.getenv("PINECONE_API_KEY"),
        ssl_verify=False,
    )
    index = pc.Index(os.getenv("INDEX_NAME"))
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    vector_store = PineconeVectorStore(
        index=index,
        embedding=embeddings,
    )
    logger.info("Ingesting documents into Pinecone...")
    vector_store.add_documents(
        documents,
    )
    logger.info(f"Successfully ingested documents into Pinecone for {path}!")


if __name__ == "__main__":
    for path in PATHS:
        ingest_docs(
            path=path.get("name"), # local folder to ingest
            link_specs=path.get("link_specs"), # link specs to convert the local folder path to a link
        )
