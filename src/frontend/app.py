import streamlit as st
import requests
from streamlit_js_eval import streamlit_js_eval

# Set the page configuration
st.set_page_config(layout="wide")

st.markdown(
    """
<style>
.stAppDeployButton {
    visibility: hidden;
}
</style>
""",
    unsafe_allow_html=True,
)

# Get the window height
screen_height = streamlit_js_eval(js_expressions="screen.height", key="HEIGHT")

# Define the backend URL
backend_url = "http://34.142.165.16:4000"


# Define the pages
def Main():
    st.title("RAG Project Portal")
    st.write(
        "Welcome to the RAG Project. This portal allows you to upload documents and ask questions to the AI."
    )

    st.write("## Services")
    st.page_link(st.Page(Add_Document), label="Add Document")
    st.page_link(st.Page(Ask), label="Ask Question")


def Add_Document():
    st.title("Add Document")
    st.write("Upload a PDF document to the backend.")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            st.write("File is ready.")
            if st.button("Upload"):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                response = requests.post(f"{backend_url}/add_document", files=files)
                if response.status_code == 200:
                    st.success("Document added successfully.")
                else:
                    st.error(f"Failed to add document: {response.json().get('error')}")
        else:
            st.error(f"Please upload a valid PDF file.: {uploaded_file.type}")


def Ask():
    st.title("Chat")

    st.sidebar.button("Reset conversation", on_click=reset_conversation)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = requests.post(
                    f"{backend_url}/ask", json={"question": prompt}
                )
                if response.status_code == 200:
                    answer = response.json().get("answer")
                    st.markdown(answer)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
                else:
                    error_message = response.json().get("error", "Failed to get answer")
                    st.error(error_message)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_message}
                    )


# Function to reset the conversation
def reset_conversation():
    """Reset the conversation by clearing the message history."""
    st.session_state.messages = []


# Define the navigation
page = st.navigation([st.Page(Main), st.Page(Add_Document), st.Page(Ask)])

page.run()
