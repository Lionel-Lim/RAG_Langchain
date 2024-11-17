def format_docs(docs):
    return [
        {"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs
    ]
