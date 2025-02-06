"""Module defining the UI of the RAG."""

import warnings
import time

import streamlit as st
from dotenv import load_dotenv

from src.backend import LLMModel, RAGChain, VectorStorePinecone

warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv("local/.env")

# Define App front page


def run_app() -> None:
    """Run the Streamlit app."""
    st.title("🦜🔗 LangChain Assistant")

    # Initialize chat session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Initialize resources
    with st.spinner("Initializing resources..."):
        retriever = VectorStorePinecone()
        llm = LLMModel(model="deepseek-r1:1.5b").llm  # deepseek-r1:1.5b deepseek-r1:14b deepseek-r1:7b
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
                search="similarity",
            )

        # Generate and stream response
        response = rag_chain.rag_chain(
            documents=documents,
            query=prompt,
            llm=llm,
        )
        
        with st.sidebar.title("Thinking process 💭"):
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
