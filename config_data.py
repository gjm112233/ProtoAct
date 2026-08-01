md5_path = "./md5.text"  # File name for storing md5 strings
collection_name = "rag"  # Vector database table name
persist_directory = "./chroma_db"  # Local folder path for vector database storage
chunk_size = 1000  # Maximum character count for each split text chunk
chunk_overlap = 100  # Allowed overlap character count between consecutive text chunks
separators = ["\n\n", "\n", ".", ",", "?", "!", "\u3002", "\uff0c", "\uff1f", "\uff01", " "]  # Natural paragraph separators
max_split_char_number = 1000  # Text splitting threshold
similarity_threshold = 5  # Number of matched documents returned by retrieval

# Fixed format: add LangChain configuration and set the session_id for the current program.
session_config = {
    "configurable": {
        "session_id": "user_001"
    }
}






