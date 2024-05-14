# Document conversion chatbot (InspiringDocs)

- [1] Installed required libraries and frameworks: [PymuPDF](https://pymupdf.readthedocs.io/en/latest/the-basics.html) for pdf read and extraction of data.

- [reddit 1st comment](https://www.reddit.com/r/MachineLearning/comments/100rbhp/d_data_cleaning_techniques_for_pdf_documents_with/)


### Why langchain framework and Faiss library? 
- [Compatibility with HuggingFace Embeddings](https://python.langchain.com/v0.1/docs/integrations/text_embedding/sentence_transformers/):
- [Huggingface Embeddings](https://api.python.langchain.com/en/latest/embeddings/langchain_community.embeddings.huggingface.HuggingFaceEmbeddings.html)
        LangChain: It seamlessly integrates with models from the HuggingFace Transformers library, such as those provided by sentence-transformers. This allows us to use pre-trained models for generating high-quality embeddings.
- LangChain and FAISS: The combination of LangChain and FAISS offers a straightforward API for adding, storing, and querying embeddings. This simplifies the development process and reduces the time required to implement the solution.

- LLamaIndex: While LLamaIndex is also a powerful framework, it may not offer the same level of integration and performance optimization specifically tailored for dense vector search as FAISS. FAISS is specifically designed for similarity search, making it more suitable for our needs.

