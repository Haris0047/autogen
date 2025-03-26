import os
import asyncio
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai
from pinecone import Pinecone
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.ui import Console
from autogen_core.memory import ListMemory

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

openai.api_key = OPENAI_API_KEY
model_client = OpenAIChatCompletionClient(
    model="gpt-4o-mini",    
    api_key=OPENAI_API_KEY,
    temperature=0.7,
)
user_memory = ListMemory()
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "autogen-rag"
index = pc.Index(index_name)

embedding_function = lambda text: openai.embeddings.create(input=text, model="text-embedding-3-small").data[0].embedding

text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "\r", "\t"], chunk_size=500, chunk_overlap=50)

def expand_query(query):
    """Generate multiple variations of the query to enhance retrieval."""
    return [
        f"Give detailed information on {query}",
        f"Explain {query} with examples",
        f"Provide recent research on {query}",
    ]

def retrieve_documents(query: str, n_results: int = 3):
    """Retrieve documents from Pinecone using multi-query expansion."""
    queries = expand_query(query)
    retrieved_docs = []
    for q in queries:
        query_embedding = embedding_function(q)
        
        results = index.query(
            vector=query_embedding,  
            top_k=n_results,        
            include_metadata=True   
        )
        
        docs = results['matches']
        for doc in docs:
            retrieved_docs.append(doc['metadata']['text'])
    
    return list(set(retrieved_docs))[:n_results]

assistant = AssistantAgent(
    name="assistant",
    model_client=model_client,
    system_message=(
        "You are a RAG-powered AI assistant. "
        "Use the `retrieve_documents` tool when a query requires external knowledge retrieval. "
        "After retrieving documents, use them to formulate an informed response."
    ),
    memory=[user_memory],
    tools=[retrieve_documents],
    reflect_on_tool_use=True,
)

confirmation = AssistantAgent(
    name="confirmation",
    model_client=model_client,
    system_message="You are a validation agent. After every response from the assistant, reply with 'APPROVE'.",
    memory=[user_memory],
)

user_proxy = UserProxyAgent("user_proxy", input_func=input)

termination = TextMentionTermination("APPROVE")

team = RoundRobinGroupChat([assistant, confirmation], termination_condition=termination)

async def interactive_chat():
    """Run interactive chat session asynchronously."""
    print("\n=== AI Assistant Chat with Multi-Query RAG ===")
    print("Type your message or 'exit' to end the conversation.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Exiting chat...")
            break

        stream = team.run_stream(task=user_input)  
        await Console(stream)

if __name__ == "__main__":
    asyncio.run(interactive_chat())