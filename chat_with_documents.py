from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import BaseRetriever
from utils import MEMORY
from langchain_pinecone import PineconeVectorStore
import os
from dotenv import load_dotenv
load_dotenv()


LLM = HuggingFaceEndpoint(
    # repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    endpoint_url="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    # temperature= 0.1,
    # repetition_penalty = 1.03,
    # max_new_tokens = 1024,
    # top_k = 30,
    huggingfacehub_api_token = os.getenv("HUGGINGFACE_API_KEY")
    )


def configure_retriever():
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
    # alternatively: 
    index_name = "legal-bot"
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-l6-v2")
    docsearch = PineconeVectorStore(index_name=index_name, embedding=embeddings, pinecone_api_key= os.getenv("PINECONE_API_KEY"))
    retriever = docsearch.as_retriever(search_kwargs = {'k':2})
    return retriever


def configure_chain(retriever: BaseRetriever):
    params = dict(
        llm=LLM,
        retriever=retriever,
        memory=MEMORY,
        verbose=True,
        max_tokens_limit=4000, 
        # prompt = prompt
        )
    
    return ConversationalRetrievalChain.from_llm(**params)


def configure_retrieval_chain():
    retriever = configure_retriever()
    chain = configure_chain(retriever=retriever)
    return chain
