# Import the libraries
import os
import PyPDF2
from pinecone import ServerlessSpec
from sentence_transformers import SentenceTransformer
import pinecone
from langchain.text_splitter import CharacterTextSplitter
import os
from dotenv import load_dotenv
load_dotenv()

# Function to read PDFs and extract text
def read_pdfs(base_path):
    all_text = []
    for i in os.listdir(os.path.join(base_path, "cases")):
        if i.lower().endswith(".pdf"):
            file_path = os.path.join(base_path, "cases", i)
            with open(file_path, 'rb') as pdf:
                reader = PyPDF2.PdfReader(pdf)
                pdf_text = ""
                for page in reader.pages:
                    pdf_text+=page.extract_text()
                all_text.append(pdf_text)
    return all_text

# Function to split text into chunks
def get_chunks(text, chunk_size=1000, chunk_overlap=200):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = splitter.split_text(text)
    return chunks

# Set Api key from .env file
pinecone_api =os.getenv("PINECONE_API_KEY")
# print(pinecone_api)

# Serverless specs for free version
cloud = 'aws'
region = 'us-east-1'
spec = ServerlessSpec(cloud=cloud, region=region)
index_name = "legal-bot"

# Create Index in Pinecone vector database
index_name = "legal-bot"
pc = pinecone.Pinecone(api_key=pinecone_api)
existing_indexes = [
    index_info["name"] for index_info in pc.list_indexes()
]

index = pc.Index(index_name)
index.describe_index_stats()

# Load the Hugging Face model for generating embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to generate embeddings for the text
def generate_embeddings(text_list):
    all_chunks = []
    for text in text_list:
        chunks = get_chunks(text)
        all_chunks.extend(chunks)
    embeddings = model.encode(all_chunks, convert_to_tensor=True)
    return embeddings, all_chunks

# Function to store embeddings in Pinecone
def store_embeddings(embeddings, chunks):
    for i, embedding in enumerate(embeddings):
        metadata = {'text': chunks[i]}
        index.upsert([(str(i), embedding.cpu().numpy().tolist(), metadata)])

# base_path = os.path.join(os.getcwd(),"drive/MyDrive")
base_path = os.getcwd()

# Read PDFs and extract text
all_text = read_pdfs(base_path)
# for i, text in enumerate(all_text):
#     print(f"PDF {i} text snippet: {text[:500]}")
print("Text extracted sucessfully!")

# Generate embeddings for the text
embeddings, chunk_texts = generate_embeddings(all_text)
print("Embedding successful!")

# Store embeddings in Pinecone
store_embeddings(embeddings, chunk_texts)
print("Embeddings stored successfully in Pinecone.")

