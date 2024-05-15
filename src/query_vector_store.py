# from langchain.chains.retrieval import create_retrieval_chain

# from transformers import GPT2Tokenizer, GPT2LMHeadModel


# class Chatbot:
#     def __init__(self,vector_store):
#         self.vector_store = vector_store
#         self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#         self.model = GPT2LMHeadModel.from_pretrained("gpt2")
    

#     def generate_response(self,query):
#         docs = self.vector_store.as_retriever(search_type='similarity',search_kwargs={'k':1}).invoke(query)
#         context = "".join(doc.page_content for doc in docs) if docs else ""

#         # Combine query and context into a prompt
#         prompt = f"Question: {query}\nContext: {context}\nAnswer:"
#         inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=1024)

#         # Generate a response using the model
#         outputs = self.model.generate(
#             inputs,
#             max_length=1024,  # Total length of input + output
#             max_new_tokens=150,  # Limit new tokens generated
#             no_repeat_ngram_size=2,
#             num_return_sequences=1,
#             pad_token_id=self.tokenizer.eos_token_id,
#             early_stopping=True
#         )
#         response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

#         return response



# if __name__ == "__main__":
#     from data_processing import ProcessDocs
#     from data_ingestion import DataIngestion    
#     processor = ProcessDocs('./data/final_paper.pdf')
#     chunks = processor.preprocess()
#     ingestion = DataIngestion(chunks)
#     vector_store = ingestion.ingest_data()
#     chatbot = Chatbot(vector_store)
#     query = "What is the maximum value of BLEU score achieved by the mt5-small model?"
#     response = chatbot.generate_response(query)
#     print(response)