"""Module containig main components of the RAG application backend."""

import os
import warnings
from typing import Generator, Literal

import streamlit as st
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from pinecone.data.index import Index
from typing_extensions import Self
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever

from src.logging import logger


warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv("local/.env")

# Constants
MODEL_NAME = "thenlper/gte-base"


class VectorStorePinecone:
    def __init__(
        self: Self,
        api_key: str = os.getenv("PINECONE_API_KEY"),
        index_name: str = os.getenv("INDEX_NAME"),
        embedding_model_name: str = MODEL_NAME,
    ) -> None:
        """
        Constructor of the VectorStorePinecone class.

        :param api_key: Pinecone API key, defaults to os.getenv("PINECONE_API_KEY")
        :type api_key: str, optional
        :param index_name: name of the Pinecone Index containing the KB, defaults to os.getenv("INDEX_NAME")
        :type index_name: str, optional
        :param embedding_model_name: name of the embedding model to be used, defaults to MODEL_NAME
        :type embedding_model_name: str, optional
        """
        self.vectorstore = PineconeVectorStore(
            index=self.get_vectorstore_index(
                api_key,
                index_name,
            ),
            embedding=self.get_embedding_model(embedding_model_name),
        )

    @staticmethod
    @st.cache_resource(show_spinner=False)
    def get_embedding_model(_model_name: str) -> HuggingFaceEmbeddings:
        """
        Static method to retrieve the embeddig model.
        It cache the model as Streamlit resources to
        avoid re-running it at each new session.

        :param _model_name: name of the HuggingFace embedding model
        :type _model_name: str
        :return: instance of the HuggingFaceEmbeddings model
        :rtype: HuggingFaceEmbeddings
        """
        logger.info(f"Loading embedding model {_model_name}...")
        return HuggingFaceEmbeddings(model_name=_model_name)

    @staticmethod
    @st.cache_resource(show_spinner=False)
    def get_vectorstore_index(
        _api_key: str,
        _index_name: str,
    ) -> Index:
        """
        Static method to retrieve the Pinecone Index.
        It cache the index as Streamlit resources to
        avoid re-running it at each new session.

        :param _api_key: Pinecone API key
        :type _api_key: str
        :param _index_name: name of the Pinecone index
        :type _index_name: str
        :return: instance of a Pinecone index
        :rtype: Index
        """
        logger.info(f"Loading Pinecone index {_index_name}...")
        pc = Pinecone(
            api_key=_api_key,
            ssl_verify=False,
        )
        return pc.Index(_index_name)

    def retrieve_documents_with_sources(
        self,
        query: str,
        search: Literal["similarity", "mmr"] = "similarity",
        top_k: int = 5,
    ) -> tuple[list[Document], set[str]]:
        """
        Retrieve relevant documents based on the search type.

        :param query: user's input query
        :type query: str
        :param search: search method (similarity or max marginal relevance)
        :type search: Literal["similarity", "mmr"]
        :param top_k: number of documents to retrieve, defaults to 5
        :type top_k: int, optional
        :return: list of retrieved documents with set of unique sources
        :rtype: tuple[list[Document], set[str]]
        """
        if search == "similarity":
            logger.info("Retrieving documents with similarity search...")
            documents = self.vectorstore.similarity_search(query=query, k=top_k)
        elif search == "mmr":
            logger.info("Retrieving documents with max marginal relevance search...")
            documents = self.vectorstore.max_marginal_relevance_search(
                query=query, k=top_k, fetch_k=(top_k * 3), lambda_mult=0.5
            )
        sources = set(doc.metadata["source"] for doc in documents)
        logger.info(f"Retrieved the following context {'\n\n'.join([document.page_content for document in documents])}")
        return documents, sources


class LLMModel:
    def __init__(
        self,
        model_name: str,
        temperature: float,
    ) -> None:
        """
        Constructor of the LLMModel class.

        :param model_name: name of the LLM (from Ollama)
        :type model_name: str
        :param temperature: temperature of the LLM
        :type temperature: float
        """
        self.llm = self.get_llm(model_name, temperature)

    @staticmethod
    @st.cache_resource(show_spinner=False)
    def get_llm(
        _model: str,
        _temperature: float,
    ) -> ChatOllama:
        """
        Static method to get an OllamaChat model.
        It cache it as Streamlit resource to avoid
        re-running it at each new session.

        :param _model: name of the LLM (Ollama) model
        :type _model: str
        :param _temperature: temperature of the LLM, defaults to 0
        :type _temperature: float, optional
        :return: instance of a ChatOllama model
        :rtype: ChatOllama
        """
        logger.info(f"Loading LLM {_model}...")
        return ChatOllama(
            model=_model,
            temperature=_temperature,
        )


class RAGChain:
    def __init__(
        self: Self,
    ) -> None:
        """
        Constructor of the RAGChain class.
        """
        template = """
        Use the following pieces of context to answer the question at the end.
        If you do not know the answer, just say that you do not know, do not try to make up an answer.
        
        <context>
        {context}
        </context>
        
        Question: {question}
        
        Answer:
        """

        self.prompt = PromptTemplate(
            template=template,
            input_variables=["question", "context"],
        )

    @staticmethod
    def format_docs(docs: list[Document]) -> str:
        """
        Format retrieved documents into a single string for context augmentation.

        :param docs: list of Langchain document objects
        :return: documents stuffed into a string and nicely spaced
        """
        return "\n\n".join([doc.page_content for doc in docs])

    def rag_chain(
        self: Self,
        documents: list[Document],
        query: str,
        llm: ChatOllama,
    ) -> Generator:
        """
        Method to steam a Runnable LangChain Chain.
        It takes retrieved documents (as LangChain Document objects),
        a query, a prompt (as attribute) and a ChatModel and stream the
        entire chain.

        :param documents: retrieved documents from a vector store
        :type documents: list[Document]
        :param query: user's input query
        :type query: str
        :param llm: instance of a ChatOllama model
        :type llm: ChatOllama
        """
        logger.info("Building RAG chain...")
        rag_chain = self.prompt | llm | StrOutputParser()
        return rag_chain.stream(
            input={"question": query, "context": self.format_docs(documents)}
        )
