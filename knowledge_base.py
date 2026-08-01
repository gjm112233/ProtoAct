"""
Knowledge base
"""

import os
from datetime import datetime
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config_data as config
import hashlib
from langchain_chroma import Chroma


# Check whether the incoming md5 string has already been processed.
# This compares it with strings stored in the md5.text file.
# Return False if it has not been processed; return True if it has been processed.
def check_md5(md5_str: str):
    if not os.path.exists(config.md5_path):
        # If ./md5.text does not exist, the current md5 string has not been processed.
        # Create the ./md5.text file.
        open(config.md5_path, 'w', encoding='utf-8').close()
        return False
    else:  # The ./md5.text file exists.
        # .readlines() gets all lines in the file and then iterates over them line by line.
        for line in open(config.md5_path, 'r', encoding='utf-8').readlines():
            line = line.strip()  # Remove leading and trailing spaces and line breaks.
            if line == md5_str:  # If a line matches the current md5_str, the uploaded file has already been processed.
                return True

        return False


# Save the incoming md5 string to the file.
def save_md5(md5_str: str):
    # Append the new string.
    with open(config.md5_path, "a", encoding='utf-8') as f:
        f.write(md5_str + "\n")


# Convert the incoming string to md5 format.
def get_string_md5(input_str: str, encoding='utf-8'):
    # First convert the string to a bytes array.
    str_bytes = input_str.encode(encoding=encoding)
    # Create an md5 object.
    md5_obj = hashlib.md5()
    # Pass in the bytes array to convert.
    md5_obj.update(str_bytes)
    # Get the hexadecimal md5 string.
    mdf_hex = md5_obj.hexdigest()
    return mdf_hex


class KnowledgeBaseService(object):
    def __init__(self):
        # Create ./chroma_db if the local vector database storage folder does not exist.
        os.makedirs(config.persist_directory, exist_ok=True)

        # Vector database instance
        self.chroma = Chroma(
            collection_name=config.collection_name,  # Name the current vector store, similar to a table name.
            # Embedding model
            embedding_function=DashScopeEmbeddings(dashscope_api_key="Your_api_key", model="text-embedding-v4"),
            persist_directory=config.persist_directory   # Folder path for data storage
        )

        # Text splitter object
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,  # Maximum character count for each chunk
            chunk_overlap=config.chunk_overlap,  # Allowed overlap character count between chunks
            # Separators used to split text into natural paragraphs
            separators=config.separators,
            length_function=len,  # Function used to count characters
        )

    # Store the incoming string in the vector database.
    def upload_by_str(self, data: str, filename):
        # First convert the incoming string to md5 format.
        md5_str = get_string_md5(data)
        if check_md5(md5_str):
            return "The content already exists in the knowledge base. Skipped."

        # Split the text only when it is larger than the threshold.
        if len(data) > config.max_split_char_number:
            # split_text() splits the raw text string and returns list(str).
            knowledge_chunks = self.spliter.split_text(data)
        else:
            # If the text is short enough, it does not need splitting, but it still needs to become list(str).
            knowledge_chunks = [data]

        meta_data = {
            "source": filename,
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator": "gu"
        }

        # Store the split text chunks in the vector database.
        self.chroma.add_texts(
            knowledge_chunks,
            metadatas=[meta_data for _ in knowledge_chunks]  # Metadata is the same for every chunk.
        )

        save_md5(md5_str)  # Save the newly added vector as md5 in the md5.text file.
        return "The content has been successfully loaded into the vector database!"


if __name__ == '__main__':
    # The same string produces the same md5 result; any small difference changes the result.
    res1 = get_string_md5("aaa")
    res2 = get_string_md5("aaa")
    res3 = get_string_md5("bbb")
    # print(res1)
    # print(res2)
    # print(res3)



