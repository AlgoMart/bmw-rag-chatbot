import streamlit as st
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import (
    Runnable,
    RunnablePassthrough,
    RunnableWithMessageHistory,
)
from langchain_openai import ChatOpenAI

from src.rag import VectorDBService

session_histories = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in session_histories:
        session_histories[session_id] = InMemoryChatMessageHistory()

    # Limit to last 5 messages (10 total: 5 human + 5 AI)
    history = session_histories[session_id]
    if len(history.messages) > 10:
        # Keep only the last 10 messages (5 exchanges)
        history.messages = history.messages[-10:]

    return history


def rag_chain() -> Runnable:
    # Use ChatPromptTemplate + MessagesPlaceholder
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Use the context provided to answer the question based on the retrieved\
                    information: {context}",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ]
    )

    # Create an LLM instance
    llm = ChatOpenAI(model="gpt-4o-mini")

    # Load the retriever
    retriever = VectorDBService().get_ventor_store_retriever(6)

    # Create the chain - simplified version that works with RunnableWithMessageHistory
    def format_context(inputs):
        """Format the context from retriever results."""
        docs = retriever.invoke(inputs["question"])
        context = "\n\n".join([doc.page_content for doc in docs])
        return {"context": context, "question": inputs["question"]}

    # Create the core chain
    chain = RunnablePassthrough.assign(context=lambda x: format_context(x)["context"]) | prompt | llm

    # Wrap the core chain with memory
    chain_with_memory = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    return chain_with_memory


# Streamlit chat interface
def chat_interface() -> None:
    """Render the chat interface for user interaction."""
    st.subheader("💬 Chat with the LLM")
    chain = rag_chain()

    # Display past messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input box for user prompt
    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Show user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate assistant response
        with st.chat_message("assistant"):
            response_container = st.empty()
            with st.spinner("Thinking..."):
                # Use the cached chain
                chain = st.session_state.rag_chain

                # Use a session id to track memory (just one for now)
                session_id = "default_session"

                # Pass the user's question as 'question' input to the chain
                response_obj = chain.invoke(
                    {"question": prompt},
                    config={"configurable": {"session_id": session_id}},
                )

                # If using ChatOpenAI, response will be a ChatMessage-like obj with .content
                response = response_obj.content if hasattr(response_obj, "content") else str(response_obj)
                st.session_state.messages.append({"role": "assistant", "content": response})

            response_container.markdown(response)


# Initialize the stramlit states
def initialize_session() -> None:
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = rag_chain()


if __name__ == "__main__":
    # Streamlit title
    st.title("BMW Chatbot")
    initialize_session()
    chat_interface()
