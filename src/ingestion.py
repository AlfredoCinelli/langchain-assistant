"""TBA."""

# Import packages and modules

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, UnstructuredHTMLLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import os
import warnings
warnings.filterwarnings("ignore")

MODEL_NAME = "thenlper/gte-base" # bi-encoder model for embeddings generation

load_dotenv("local/.env")
PATH = "langchain_api_docs" # scraped Langchain docs

def source_to_link(
    source: str,
    local_folder: str = "langchain_api_docs/",
    prefix: str = "https://python.langchain.com/api_reference/",
) -> str:
    """
    Convert a source string to a link.
    
    :param source: source of the file from local FS
    :type source: str
    :param local_folder: local folder where the docs are stored, defaults to "langchain-docs/"
    :type local_folder: str, optional
    :param prefix: prefix of the original web page, defaults to "https://python.langchain.com/api_reference/"
    :type prefix: str, optional
    :return: link to the source on the original web page
    :rtype: str
    """
    return source.replace(local_folder, prefix)

def ingest_docs():
    print("Loading documents...")
    loader = DirectoryLoader(
        path=PATH,
        glob="**/*.html",
        loader_cls=UnstructuredHTMLLoader,
    )
    raw_docs = loader.load()
    print(f"loaded {len(raw_docs)} documents")
    
    print("Splitting documents...")
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=400,
        chunk_overlap=40,
    )
    
    documents = splitter.split_documents(raw_docs)
    print("Updating documents...")
    for doc in documents:
        source_url = source_to_link(source=doc.metadata["source"])
        doc.metadata.update({"source": source_url})
        
    print(f"Index {len(documents)} documents")
    
    print("Initializing Pinecone...")
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
    print("Ingesting documents into Pinecone...")
    vector_store.add_documents(
        documents,
    )
    print("Done!")


if __name__ == "__main__":
    ingest_docs()
