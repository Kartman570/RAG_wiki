import os
import pandas as pd

from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from datasets import load_dataset

index_dir = "./faiss_index/"
dataset_dir = "./dataset/"

# ~8.2 GB RAM needed to index dataset
dataset = load_dataset("wikipedia", "20220301.simple", split="train", cache_dir=dataset_dir)
embeddings = HuggingFaceEmbeddings()


def get_index():
    if os.path.exists(index_dir) and os.path.isdir(index_dir):
        files = os.listdir(index_dir)
        if files:
            db = FAISS.load_local("./faiss_index", embeddings)
            return db
    df = pd.DataFrame(dataset)
    df.head()
    loader = DataFrameLoader(df, page_content_column='title')
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    db = FAISS.from_documents(texts, embeddings)
    db.as_retriever()
    db.save_local('faiss_index')
    return db


db = get_index()
hf = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 100},
)
template = """Answer a question using only given context. 
If you can not provide answer only by using given context, say 'I don`t know'.
Keep your answer short - one sentence max. Type ONLY answer itself, do not type 'answer is' or commentaries.

CONTEXT: {context}
QUESTION: {question}
ANSWER: 
"""


prompt = PromptTemplate(
    input_variables=["context", "question"], template=template
)
chain = LLMChain(
    llm=hf,
    prompt=prompt)


def main(question: str):
    relevants = db.similarity_search(question)
    doc = relevants[0].dict()['metadata']['text']
    doc = doc.replace('\n\n','\n')
    answer = chain.invoke({
        "context": doc,
        "question": question
    })
    print(f"ANSWER IS:\n{answer['text']}")
    return answer['text']


if __name__ == "__main__":
    question_manual = input("What's your question? ")
    main(question_manual)
