import streamlit as st
import requests
import pandas as pd
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
backend_url = "http://backend_service:4000/"


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
    st.title("Document Management")

    # Create two tabs: one for uploading documents and one for viewing document information
    tab1, tab2 = st.tabs(["Upload Document", "Document Information"])

    with tab1:
        st.header("Upload Document")
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
                st.error(f"Please upload a valid PDF file: {uploaded_file.type}")




    with tab2:
        st.header("Document Information")

        # Mock data for testing
        mock_data = [
            {"Document Name": "Document1.pdf", "Uploaded Date": "2023-12-01", "GUID": "123-abc", "File Size (KB)": 245, "Pages": 10},
            {"Document Name": "Document2.pdf", "Uploaded Date": "2023-12-02", "GUID": "456-def", "File Size (KB)": 567, "Pages": 20},
            {"Document Name": "Document3.pdf", "Uploaded Date": "2023-12-03", "GUID": "789-ghi", "File Size (KB)": 789, "Pages": 15}
        ]
        df = pd.DataFrame(mock_data)

        st.write("Uploaded Documents")

        # Create a selectbox for the user to choose a document
        selected_document = st.selectbox("Select a document", df["Document Name"].tolist())

        # Display the details of the selected document
        if selected_document:
            document_details = df[df["Document Name"] == selected_document].iloc[0]
            st.subheader(f"Details for {selected_document}")
            st.write(f"GUID: {document_details['GUID']}")
            st.write(f"File Size: {document_details['File Size (KB)']} KB")
            st.write(f"Number of Pages: {document_details['Pages']}")
            st.write(f"Uploaded Date: {document_details['Uploaded Date']}")


def Ask():
    st.title("Chat")

    # Reset conversation button
    st.sidebar.button("Reset conversation", on_click=reset_conversation)

    # Fetch document list from backend (replace with actual API call)
    documents = [
        {"name": "Document1.pdf", "guid": "123-abc"},
        {"name": "Document2.pdf", "guid": "456-def"},
        {"name": "Document3.pdf", "guid": "789-ghi"}
    ]

    # Create multi-select dropdown
    selected_docs = st.sidebar.multiselect(
        "Select documents for search",
        options=[doc["name"] for doc in documents],
        format_func=lambda x: x
    )

    # Display selected documents
    if selected_docs:
        st.sidebar.write("Selected documents:")
        st.sidebar.write(f"{len(selected_docs)} {"document" if len(selected_docs) == 1 else "documents"} selected.")
    else:
        st.sidebar.write("No specific documents selected. \n\n Searching all documents.")

    # Get GUIDs of selected documents
    selected_guids = [doc["guid"] for doc in documents if doc["name"] in selected_docs]

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
                # Prepare the payload
                payload = {"question": prompt}
                if selected_guids:
                    payload["document_guids"] = selected_guids

                response = requests.post(f"{backend_url}/ask", json=payload)
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
