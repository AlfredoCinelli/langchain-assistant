"""TBA."""

# Import packages and modules

from dotenv import load_dotenv
from langchain_community.document_loaders import FireCrawlLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import os
import warnings
warnings.filterwarnings("ignore")

MODEL_NAME = "thenlper/gte-base" # bi-encoder model for embeddings generation

load_dotenv("local/.env")

LANGCHAIN_DOCS_LINKS = [
    "https://python.langchain.com/docs/how_to/", # how to secion of LangChain doc
    "https://python.langchain.com/docs/concepts/", # concepts section of LangChain doc
    "https://python.langchain.com/docs/tutorials/", # tutorials section of LangChain doc
]

def ingest_docs():
    print("Initializing Pinecone...")
    pc = Pinecone(
        api_key=os.getenv("PINECONE_API_KEY"),
        ssl_verify=False,
    )
    index = pc.Index(os.getenv("FIRECRAWL_INDEX_NAME"))
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    vector_store = PineconeVectorStore(
        index=index,
        embedding=embeddings,
    )
    print("Start crawling...")
    for url in LANGCHAIN_DOCS_LINKS:
        print(f"Crawling with FireCrawl link: {url}")
        loader = FireCrawlLoader(
            url=url,
            mode="crawl",
            params={
                "limit": 150, # maximum number of pages to scrape
                "scrapeOptions": {
                    "formats": ["markdown", "html"], # formats to include in the response
                    "onlyMainContent": True, # the scraper returns the main content only without headers, navigation bar, footers and so on
                },
                "maxDepth": 5, # maximum scraping recursion relative to the supplied URL
                #"wait_until_done": True, # get the result only after the crawl job is done
            }
        )
        documents = loader.load()

        print(f"Index {len(documents)} documents")
    
        print("Ingesting documents into Pinecone...")
        vector_store.add_documents(
            documents,
        )
    print("Done!")


if __name__ == "__main__":
    ingest_docs()
