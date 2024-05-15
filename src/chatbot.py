from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
import torch
from langchain.memory import ConversationBufferMemory
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts.prompt import PromptTemplate
from data_processing import ProcessDocs
from data_ingestion import DataIngestion
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline


class Chatbot:
    def __init__(self,pdf_path):
        self.pdf_path = pdf_path
        self.setup_model()
        self.setup_vector_store()
        self.setup_chat_components()

    def setup_model(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.pipe = pipeline("text-generation",
                             model=self.model,
                             tokenizer=self.tokenizer,
                             config={"max_length": 512})
        self.llm = HuggingFacePipeline(pipeline=self.pipe)


    def setup_vector_store(self):
        """Process documents and setup the vector store."""
        processor = ProcessDocs(self.pdf_path)
        chunks = processor.preprocess()
        ingestion = DataIngestion(chunks)
        self.vector_store = ingestion.ingest_data()
        
    def setup_chat_components(self):
        """Setup retrieval chain and memory components for the chatbot."""
        self.memory = ConversationBufferMemory(input_key='context', memory_key='history', return_messages=True)
        self.prompt_template = PromptTemplate(template="""
### System:
You are an AI assistant that follows instructions extremely well. Help as much as you can.
### User:
You are a research assistant for an artificial intelligence student. Use only the following information to answer user queries:
Context= {context}
History = {history}
Question= {question}
### Assistant:
""", input_variables=["context", "history", "question"])
        
        retriever = self.vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 1})
        question_answer_chain = create_stuff_documents_chain(self.llm, self.prompt_template)
        self.chain = create_retrieval_chain(retriever, question_answer_chain)

    def chat(self, query):
        # Using async to handle coroutine for abuffer_as_str
        context = self.memory.abuffer_as_str()  # Now properly awaited
        history = self.memory.buffer_as_str  # Accessed as a property, not a method
        input_data = {
            "context": context,
            "history": history,
            "question": query
        }
        response = self.chain.invoke(input_data)
        self.memory.save_context({"question": query}, {"response": response})
        return response
if __name__ == "__main__":
    chatbot = Chatbot(pdf_path='./data/final_paper.pdf')
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        response = chatbot.chat(user_input)
        print("AI:", response)