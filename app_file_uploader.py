"""
Building a Web File Upload Service with Streamlit
"""

import time
import streamlit as st
from knowledge_base import KnowledgeBaseService

# Add the page title
st.title("Knowledge Base Update Service")

# file_uploader()  File Upload Function
upload_file = st.file_uploader(
    "Please upload a txt file",
    type=['txt'],
    accept_multiple_files=False,  # Whether to accept multiple file uploads. False indicates that only one file can be uploaded.
)

# the Session State Recorder is a dictionary that can store variables whose state needs to be preserved, the variables stored within it will not be reset upon page refresh.
# st.session_state
if "service" not in st.session_state:
    # Create a service instance on the first run
    st.session_state["service"] = KnowledgeBaseService()

if upload_file is not None:
    # Extract File Information
    file_name = upload_file.name
    file_type = upload_file.type
    file_size = upload_file.size / 1024  # KB

    st.subheader(f"File Name:{file_name}")
    st.write(f"File Format:{file_type},File Size:{file_size: .2f} KB")

    # getvalue -> bytes -> decode('utf-8')
    text = upload_file.getvalue().decode('utf-8')

    # A loading spinner is displayed while the code inside the spinner is being executed.
    with st.spinner("Loading into the knowledge base..."):
        time.sleep(1)
        # Store the uploaded text in the vector database
        res = st.session_state["service"].upload_by_str(text, file_name)
        st.write(res)  # Display the content on the page

