"""Module defining the UI of the RAG."""

import warnings
import time

import streamlit as st
from dotenv import load_dotenv

from src.backend import LLMModel, RAGChain, VectorStorePinecone
from src.logging import logger

warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv("local/.env")

# Define App utils

def create_llm_config_form() -> None:
    """
    Creates a Streamlit form for configuring LLM settings.

    It allows to configure:
    * The DeepSeek-R1 model to be used
    * The number of most relevant documents to retrieve
    * The search method to be used
    * The temperature to be used
    """
    # Initialize session state if not already done
    if "config_submitted" not in st.session_state:
        st.session_state.config_submitted = False
        st.session_state.model_name = None
        st.session_state.search = None
        st.session_state.top_k = None
        st.session_state.temperature = None

    # Available models DeepSeek-R1 models on Ollama
    available_models = [
        "deepseek-r1:1.5b",
        "deepseek-r1:7b",
        "deepseek-r1:8b",
    ]
    if st.session_state.get("config_submitted") is False:
        with st.form("llm_config"):
            st.write("## 🎛️ LLM Configurations")

            # Model selection
            model_name = st.selectbox(
                "Select DeepSeek-R1 model",
                options=available_models,
                help="Choose the DeepSeek-R1 flavour to use",
                index=None,
                placeholder="Select a model",
            )

            # Search approach
            search = st.selectbox(
                "Select Search Approach",
                options=["similarity", "mmr"],
                help="Choose the search approach to use",
                placeholder="Select a search approach",
                index=None,
            )

            # Top K setting
            top_k = st.slider(
                "Top K for Retrieval",
                min_value=1,
                max_value=20,
                value=5,
                step=1,
                help="Number of most relevant documents to retrieve",
            )

            # Additional settings could be added here
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.1,
                help="Controls randomness in the output (0 = deterministic, 1 = creative)"
            )
        
            # Submit button
            submitted = st.form_submit_button("Apply Settings", help="Apply the selected settings")
            logger.info(f"Chosen model: {model_name}, search: {search}, top_k: {top_k}, temperature: {temperature}")
            
            if submitted is True:
                st.session_state.config_submitted = True
                st.session_state.model_name = model_name
                st.session_state.search = search
                st.session_state.top_k = top_k
                st.session_state.temperature = temperature
                st.success(f"Settings applied!\n\n Using '{model_name}' with {top_k} retrieved documents and '{search}' search", icon="✅")
                time.sleep(3)
                st.rerun()  # Rerun the app to refresh the page

# Define App front page

st.set_page_config(page_title="Assistant")
def run_app() -> None:
    """Run the Streamlit app."""
    st.title("🦜🔗 LangChain Assistant")
    
    # Get App config via form
    if not st.session_state.get("config_submitted"):
        logger.info("Configuring LLM")
        create_llm_config_form()

    # Initialize chat session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Initialize resources
    with st.spinner("Initializing resources..."):
        retriever = VectorStorePinecone()
        llm = LLMModel(
            model_name=st.session_state.model_name,
            temperature=st.session_state.temperature,
        ).llm
        rag_chain = RAGChain()

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        role, content = message["role"], message["content"]
        avatar = "🧑‍💻" if role == "user" else "🤖"
        with st.chat_message(role, avatar=avatar):
            st.markdown(content)

    # User input
    if prompt := st.chat_input("Ask a question about LangChain"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)

        # Retrieve documents and create RAG chain
        with st.spinner("⚙️ Retrieving relevant documents..."):
            documents, sources = retriever.retrieve_documents_with_sources(
                query=prompt,
                search=st.session_state.search,
                top_k=st.session_state.top_k,
            )
            logger.info(f"Retrieved {len(documents)} documents")

        # Generate and stream response
        response = rag_chain.rag_chain(
            documents=documents,
            query=prompt,
            llm=llm,
        )
        
        with st.sidebar.title("💭 Thinking process"):
            thinking_process = ""
            for chunk in response:
                if "<think" in chunk:
                    continue
                elif "</think" in chunk:
                    break
                thinking_process += chunk
                st.markdown(thinking_process + "▌")
        with st.spinner("🗣️ Generating answer..."):
            with st.chat_message("assistant", avatar="🤖"):
                response_placeholder = st.empty()
                full_response = ""
                inside_think_phase = False
                for chunk in response:
                    if "<think>" in chunk:
                        inside_think_phase = True
                        continue
                    
                    if "</think>" in chunk:
                        inside_think_phase = False
                        continue
                    
                    if inside_think_phase is False:
                        full_response += chunk
                        response_placeholder.markdown(full_response + "▌")
            st.success("Successfully generated the answer!", icon="✅")

        # Add sources to the message in an expander
        with st.expander("Sources", expanded=False):
            st.caption("\n".join(f"- {source}" for source in sources))

        # Add assistant response to chat history
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_response,
            }
        )


if __name__ == "__main__":
    run_app()
