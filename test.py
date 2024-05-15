from langchain_core.prompts.prompt import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import torch 
from data_processing import ProcessDocs
from data_ingestion import DataIngestion

processor = ProcessDocs('./data/final_paper.pdf')
chunks = processor.preprocess()
ingestion = DataIngestion(chunks)
vector_store = ingestion.ingest_data()
# print(vector_store.similarity_search("llama2",k=1))
retriever = vector_store.as_retriever(search_type='similarity',search_kwargs={'k':1})

# retrieved_relevant_docs = retriever.get_relevant_documents(
    # "what is QLORA?"
# )
# 
# print(retrieved_relevant_docs)    
# 

custom_prompt_template = """
### System:
You are an AI assistant that follows instructions extremely well. Help as much as you can.
### User:
You are a research assistant for an artificial intelligence student. Use only the following information to answer user queries:
Context= {context}
History = {history}
Question= {question}
### Assistant:
"""


quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    
)
model_id = "Deci/DeciLM-7B-instruct" # model repo id
device = 'cuda'

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id,
                                             trust_remote_code=True,
                                             device_map = "auto",
                                          quantization_config=quant_config)


# create a pipeline
pipe = pipeline("text-generation",
                model=model,
                tokenizer=tokenizer,
                temperature=1e-3,
                return_full_text = False,
                max_new_tokens=2048)
llm = HuggingFacePipeline(pipeline=pipe)


prompt_template = PromptTemplate(template = custom_prompt_template,input_variables=["context","history","question"])
memory = ConversationBufferMemory(input_key='context',memory_key='history',return_messages=True)

question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
chain = create_retrieval_chain(retriever, question_answer_chain)

response = chain.invoke({"input":"What is QLORA?"})
print(response)
