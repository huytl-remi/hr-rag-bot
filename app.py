import streamlit as st
from openai import OpenAI
import tempfile
import os
import time
from typing import Dict, List

# Load the OpenAI API key securely from Streamlit secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

class RAGBot:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.init_session_state()

    def init_session_state(self):
        """Initialize session state variables."""
        default_state = {
            "messages": [],
            "assistant": None,
            "vector_store_id": None,
            "thread_id": None,
        }
        for key, value in default_state.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def create_vector_store(self) -> str:
        """Create a new vector store for file search."""
        try:
            vector_store = self.client.beta.vector_stores.create(name="HR Docs Vector Store")
            st.session_state.vector_store_id = vector_store.id
            return vector_store.id
        except Exception as e:
            st.error(f"Failed to create vector store: {str(e)}")
            return None

    def upload_files_to_vector_store(self, file_paths: List[str]) -> bool:
        """Uploads files to the vector store and polls for completion."""
        vector_store_id = st.session_state.vector_store_id
        if not vector_store_id:
            st.error("Vector store ID not found.")
            return False

        file_streams = []
        try:
            for file_path in file_paths:
                file_streams.append(open(file_path, "rb"))

            file_batch = self.client.beta.vector_stores.file_batches.upload_and_poll(
                vector_store_id=vector_store_id, files=file_streams
            )
            return file_batch.status == "completed"
        except Exception as e:
            st.error(f"File upload failed: {str(e)}")
            return False
        finally:
            # Close and delete temporary files
            for fs in file_streams:
                fs.close()
                os.unlink(fs.name)

    def create_assistant(self) -> str:
        """Creates an assistant linked to the vector store if not already created."""
        if st.session_state.assistant:
            return st.session_state.assistant

        if not st.session_state.vector_store_id:
            st.error("Vector store not available. Upload files first.")
            return None

        try:
            assistant = self.client.beta.assistants.create(
                name="HR Assistant",
                description="Answers HR-related questions based on uploaded documents",
                model="gpt-4o-mini",
                instructions="Provide helpful responses without repeating the question.",
                tools=[{"type": "file_search"}],
                tool_resources={"file_search": {"vector_store_ids": [st.session_state.vector_store_id]}},
            )
            st.session_state.assistant = assistant.id
            return assistant.id
        except Exception as e:
            st.error(f"Failed to create assistant: {str(e)}")
            return None

    def create_thread(self) -> str:
        """Creates a new thread or reuses an existing one."""
        if st.session_state.thread_id:
            return st.session_state.thread_id

        try:
            thread = self.client.beta.threads.create()
            st.session_state.thread_id = thread.id
            return thread.id
        except Exception as e:
            st.error(f"Failed to create thread: {str(e)}")
            return None

    def add_user_message(self, thread_id: str, content: str) -> None:
        """Adds a user message to the thread."""
        try:
            # Save user message to session state first
            st.session_state.messages.append({"role": "user", "content": content})

            self.client.beta.threads.messages.create(
                thread_id=thread_id, role="user", content=content
            )
        except Exception as e:
            st.error(f"Failed to add user message: {str(e)}")

    def get_assistant_response(self, thread_id: str) -> Dict:
        """Retrieves the assistant's latest response from the thread."""
        if not st.session_state.assistant:
            self.create_assistant()
            if not st.session_state.assistant:
                return {"error": "Assistant not available."}

        try:
            run = self.client.beta.threads.runs.create(
                thread_id=thread_id, assistant_id=st.session_state.assistant
            )
            # Poll for run completion
            while run.status not in ["completed", "failed"]:
                time.sleep(1)
                run = self.client.beta.threads.runs.retrieve(
                    thread_id=thread_id, run_id=run.id
                )

            if run.status == "completed":
                # Retrieve all messages in the thread
                messages = self.client.beta.threads.messages.list(thread_id=thread_id)
                # Filter to get only the most recent assistant message
                latest_response = None
                for message in reversed(messages.data):
                    if message.role == "assistant":
                        # Check if we've already processed this message
                        if st.session_state.get("last_message_id") != message.id:
                            latest_response = message.content[0].text.value
                            st.session_state["last_message_id"] = message.id  # Update last message ID
                            break

                if latest_response:
                    # Save the assistant's new response to session state
                    st.session_state.messages.append(
                        {"role": "assistant", "content": latest_response}
                    )
                    return {"answer": latest_response}
                else:
                    return {"error": "No new assistant response found."}
            else:
                return {"error": f"Assistant run failed: {run.status}"}
        except Exception as e:
            return {"error": str(e)}  # Return the error message for display

# Helper functions for UI and bot interaction
def handle_file_upload(bot):
    """Handles the file upload process and vector store creation."""
    uploaded_files = st.sidebar.file_uploader(
        "Upload HR documents",
        type=["pdf", "txt", "doc", "docx"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        file_paths = []
        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=file.name) as tmp_file:
                tmp_file.write(file.getvalue())
                file_paths.append(tmp_file.name)

        with st.spinner("Uploading documents..."):
            if not st.session_state.vector_store_id:
                bot.create_vector_store()
            if bot.upload_files_to_vector_store(file_paths):
                st.success("Files uploaded and vector store created successfully.")
            else:
                st.error("Failed to upload files to vector store.")

def display_chat_history():
    """Displays the chat history between user and assistant."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def main():
    st.set_page_config(
        page_title="HR Knowledge Base Assistant",
        page_icon="📚",
        layout="wide",
    )
    st.title("📚 HR Knowledge Base Assistant")

    # Initialize bot
    bot = RAGBot()

    # Document upload section
    with st.sidebar:
        st.header("📁 Document Upload")
        handle_file_upload(bot)
        st.divider()
        st.markdown(
            """
            ### Instructions:
            1. Upload HR documents
            2. Ask questions about the documents
            3. See answers
            """
        )

    # Chat input
    prompt = st.chat_input("Ask about your HR documents")
    if prompt:
        # Input validation
        if prompt.strip() == "":
            st.error("Please enter a valid question.")
        else:
            # Create or reuse a thread
            thread_id = bot.create_thread()
            if thread_id:
                bot.add_user_message(thread_id, prompt)
                with st.spinner("Assistant is typing..."):
                    response = bot.get_assistant_response(thread_id)
                    if "error" in response:
                        st.error(response["error"])
            else:
                st.error("Failed to create or retrieve thread.")

    # Display existing messages in the chat history
    display_chat_history()

if __name__ == "__main__":
    main()
