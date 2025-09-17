import uuid

import streamlit as st
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

from src.rag import VectorDBService


class BMWChatbot:
    SESSION_HISTORIES = {}
    RAG_CHAIN_SESSION_STATE = "rag_chain"
    CHAT_SESSION_STATE = "chat_sessions"
    CHAT_TITLE_SESSION_STATE = "chat_titles"
    CURRENT_CHAT_ID_SESSION_STATE = "current_chat_id"

    # Initialize Session
    def initialize_session(self) -> None:
        if BMWChatbot.RAG_CHAIN_SESSION_STATE not in st.session_state:
            st.session_state.rag_chain = self.rag_chain()
        if BMWChatbot.CHAT_SESSION_STATE not in st.session_state:
            st.session_state.chat_sessions = {}
        if BMWChatbot.CHAT_TITLE_SESSION_STATE not in st.session_state:
            st.session_state.chat_titles = {}
        if BMWChatbot.CURRENT_CHAT_ID_SESSION_STATE not in st.session_state:
            default_id = str(uuid.uuid4())[:8]
            st.session_state.current_chat_id = default_id
            st.session_state.chat_sessions[default_id] = []
            st.session_state.chat_titles[default_id] = "New Chat"

    # Session Management
    def get_session_history(self, session_id: str) -> InMemoryChatMessageHistory:
        """Manages memory for each session id"""
        if session_id not in BMWChatbot.SESSION_HISTORIES:
            BMWChatbot.SESSION_HISTORIES[session_id] = InMemoryChatMessageHistory()
        # keep last 10 messages only
        if len(BMWChatbot.SESSION_HISTORIES[session_id].messages) > 10:
            BMWChatbot.SESSION_HISTORIES[session_id].messages = BMWChatbot.SESSION_HISTORIES[session_id].messages[-10:]
        return BMWChatbot.SESSION_HISTORIES[session_id]

    # RAG Chain
    def rag_chain(self) -> RunnableWithMessageHistory:
        # Use ChatPromptTemplate + MessagesPlaceholder
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant. Use the context provided to answer the question based on\
                        the retrieved information: {context}",
                ),
                MessagesPlaceholder("chat_history"),
                ("human", "{question}"),
            ]
        )

        # Create an LLM instance
        llm = ChatOpenAI(
            base_url="http://localhost:12434/engines/v1",
            model="ai/llama3.2",
            api_key="not needed",
        )

        # Load the retriever
        retriever = VectorDBService().get_ventor_store_retriever(6)

        # Create the chain - simplified version that works with RunnableWithMessageHistory
        def format_context(inputs: dict) -> dict:
            """Format the context from retriever results."""
            docs = retriever.invoke(inputs["question"])
            context = "\n\n".join([doc.page_content for doc in docs])
            return {"context": context, "question": inputs["question"]}

        # Create the core chain
        chain = RunnablePassthrough.assign(context=lambda x: format_context(x)["context"]) | prompt | llm
        return RunnableWithMessageHistory(
            chain,
            self.get_session_history,
            input_messages_key="question",
            history_messages_key="chat_history",
        )

    # Sidebar Config with Delete Option
    def sidebar_config(self) -> None:
        st.sidebar.header("🛠️ Chat Options")

        if BMWChatbot.CHAT_SESSION_STATE not in st.session_state:
            st.session_state.chat_sessions = {}
        if BMWChatbot.CHAT_TITLE_SESSION_STATE not in st.session_state:
            st.session_state.chat_titles = {}

        # New Chat button
        if st.sidebar.button("✨ New Chat", use_container_width=True):
            new_id = str(uuid.uuid4())[:8]
            st.session_state.current_chat_id = new_id
            st.session_state.chat_sessions[new_id] = []
            st.session_state.chat_titles[new_id] = "New Chat"
            st.rerun()

        st.sidebar.markdown("### 💬 Conversations")

        # Show chat titles with delete option
        for cid in list(st.session_state.chat_sessions.keys()):
            cols = st.sidebar.columns([0.8, 0.2])  # left: title, right: delete
            title = st.session_state.chat_titles.get(cid, "New Chat")

            with cols[0]:
                if st.button(title, key=f"btn_{cid}", use_container_width=True):
                    st.session_state.current_chat_id = cid
                    st.rerun()

            with cols[1]:
                if st.button("❌", key=f"del_{cid}"):
                    # delete chat
                    del st.session_state.chat_sessions[cid]
                    del st.session_state.chat_titles[cid]
                    if st.session_state.current_chat_id == cid:
                        if st.session_state.chat_sessions:
                            st.session_state.current_chat_id = next(iter(st.session_state.chat_sessions.keys()))
                        else:
                            new_id = str(uuid.uuid4())[:8]
                            st.session_state.current_chat_id = new_id
                            st.session_state.chat_sessions[new_id] = []
                            st.session_state.chat_titles[new_id] = "New Chat"
                    st.rerun()

        # Clear All button
        if st.sidebar.button("🗑️ Delete All Chats", use_container_width=True):
            st.session_state.chat_sessions.clear()
            st.session_state.chat_titles.clear()
            default_id = str(uuid.uuid4())[:8]
            st.session_state.current_chat_id = default_id
            st.session_state.chat_sessions[default_id] = []
            st.session_state.chat_titles[default_id] = "New Chat"
            st.rerun()

    # Chat Interface
    def chat_interface(self) -> None:
        st.title("🚗 BMW Chatbot")
        current_id = st.session_state.current_chat_id
        messages = st.session_state.chat_sessions.get(current_id, [])

        # Display previous conversation
        for message in messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # User input
        if prompt := st.chat_input("Ask me anything about BMW vehicles..."):
            # Save the user message
            messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # LLM assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    chain = st.session_state.rag_chain
                    response_obj = chain.invoke(
                        {"question": prompt},
                        config={"configurable": {"session_id": current_id}},
                    )
                    response = response_obj.content if hasattr(response_obj, "content") else str(response_obj)
                st.markdown(response)
                messages.append({"role": "assistant", "content": response})

            # Update chat title after first reply
            if st.session_state.chat_titles.get(current_id) == "New Chat":
                title = prompt[:25] + ("..." if len(prompt) > 25 else "")
                st.session_state.chat_titles[current_id] = title

            # Save messages back
            st.session_state.chat_sessions[current_id] = messages
            st.rerun()

    # Main
    def main(self):
        st.set_page_config(page_title="BMW Chatbot", page_icon="🚗", layout="wide")
        self.initialize_session()
        self.sidebar_config()
        self.chat_interface()


if __name__ == "__main__":
    BMWChatbot().main()
