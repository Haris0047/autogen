import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import fitz  # PyMuPDF
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

load_dotenv()

# Replace with your actual API key
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY
# Create an instance of the Pinecone class
pc = Pinecone(api_key=PINECONE_API_KEY)

# Connect to your Pinecone index
index_name = "autogen-rag"
index = pc.Index(index_name)

# Load the PDF
pdf_path = 'rag.pdf'  # Path to your PDF file

# Function to extract text from the PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

# Extract text from the PDF
pdf_text = extract_text_from_pdf(pdf_path)

# Split the text into smaller chunks (optional but recommended for large documents)
# You can split based on sentences or paragraphs
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=80,
    length_function=len,
    is_separator_regex=False,
)

semantic_chunker = SemanticChunker(OpenAIEmbeddings(), breakpoint_threshold_type="percentile",)
documents = semantic_chunker.create_documents([pdf_text])
# print(documents)
chunks = text_splitter.split_text(pdf_text)
# print("=======================================================")
# print(chunks)
# exit(0)
# Generate embeddings for each chunk of text
def generate_embeddings(chunks):
    embeddings = []
    try:
        # Loop through each chunk and generate embeddings
        for chunk in documents:
            response = openai.embeddings.create(
                model="text-embedding-3-small",  # Updated embedding model
                input=chunk.page_content
            )
            # print(response.data[0].embedding)
            # break
            embeddings.append(response.data[0].embedding)  # Collect the embeddings
    except Exception as e:
        print(f"Error generating embeddings: {e}")
    return embeddings

# Generate embeddings for the PDF text chunks
embeddings = generate_embeddings(chunks)

print(f"Total chunks: {len(chunks)}")
print(f"Total embeddings: {len(embeddings)}")

# Insert the embeddings into Pinecone (along with metadata)
vectors = [(str(i), embeddings[i], {'text': documents[i].page_content}) for i in range(len(documents))]

# Upsert vectors into Pinecone index
index.upsert(vectors=vectors)

# Verify success
print(f"PDF content inserted successfully into {index_name} index.")
