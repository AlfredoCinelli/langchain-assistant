"""Module defining the UI of the RAG."""

import warnings

import streamlit as st
from dotenv import load_dotenv

from src.backend import RAGChain2

warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv("local/.env")

# Define App front page


def main():
    st.title("🦜🔗 LangChain Assistant")

    # Initialize chat session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "resources" not in st.session_state:
        st.session_state.resources = None

    # Initialize resources
    if st.session_state.resources is None:
        with st.spinner("Initializing resources..."):
            rag_chain = RAGChain2()
        st.session_state.resources = rag_chain

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

        # Generate and stream response
        with st.spinner("Generating answer..."):
            rag_chain = st.session_state.resources
            response = rag_chain.rag_chain(
                query=prompt,
                chat_history=st.session_state.messages,
            )
            with st.chat_message("assistant", avatar="🤖"):
                response_placeholder = st.empty()
                full_response = ""

                # Stream the response
                for chunk in response:
                    if "answer" in chunk:
                        full_response += chunk.get("answer")
                        response_placeholder.markdown(full_response + "▌")
                    elif "context" in chunk:
                        sources = set(doc.metadata["source"] for doc in chunk.get("context"))


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
    main()