from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.docstore import InMemoryDocstore
import numpy as np
from collections import defaultdict
import faiss


class DataIngestion:
    def __init__(self,chunks):
        self.chunks = chunks
        self.embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
        self.index = faiss.IndexFlatL2(384) #type of faiss index
        self.docstore = InMemoryDocstore()
        self.index_to_docstore_id = defaultdict(str)

        self.vector_store = FAISS(
            embedding_function = self.embedding_model,
            index = self.index,
            docstore = self.docstore,
            index_to_docstore_id = self.index_to_docstore_id
        )
    
    def create_embeddings(self):
        # embeddings = [self.embedding_model.encode(chunk) for chunk in self.chunks]
        embeddings = self.embedding_model.embed_documents(self.chunks)
        # print(embeddings[0])
        return embeddings

    def ingest_data(self):
        embeddings = self.create_embeddings()
        for i,embedding in enumerate(embeddings):
            self.vector_store.add_embeddings([(self.chunks[i], embedding)])
        
        return self.vector_store

# if __name__ == "__main__":
#     # Process and ingest data
#     from data_processing import ProcessDocs

#     processor = ProcessDocs('./data/final_paper.pdf')
#     chunks = processor.preprocess()
#     ingestion = DataIngestion(chunks)
#     vector_store = ingestion.ingest_data()
#     # print(vector_store.similarity_search("llama2",k=1))
#     retriver = vector_store.as_retriever(search_type='similarity',search_kwargs={'k':2})
#     docs = retriver.invoke("what different approaches were used for finetuning?")
    # print(docs)
    # Query the vector store
    



