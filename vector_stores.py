"""
Get the vector store retriever.
"""

from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
import config_data as config


class VectorStoreService(object):
    def __init__(self):
        self.vector_store = Chroma(
            collection_name=config.collection_name,  # Name the current vector store, similar to a table name.
            # Embedding model
            embedding_function=DashScopeEmbeddings(dashscope_api_key="Your_api_key", model="text-embedding-v4"),
            persist_directory=config.persist_directory   # Folder path for data storage
        )

    # Return the vector retriever so it can be added to the chain.
    def get_retriever(self):
        retriever = self.vector_store.as_retriever(search_kwargs={"k": config.similarity_threshold})
        return retriever


if __name__ == '__main__':
    service = VectorStoreService()
    retriever = service.get_retriever()
    documents = retriever.invoke("biological experiment parsing")
    # print(documents)
    for document in documents:
        print(document.page_content)
        print("========================================")




