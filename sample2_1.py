import os
import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken

# Define placeholder tool functions.
def text_sqdb(query: str) -> str:
    # In a real scenario, implement logic to convert text to SQL.
    return f"SQL Query generated from '{query}'"

def text_vcdb(query: str) -> str:
    # In a real scenario, implement logic to interact with a vector database.
    return f"Vector DB result for query '{query}'"

# Initialize the OpenAI model client.
model_client = OpenAIChatCompletionClient(
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY"),
    seed=42,
    temperature=0
)

# Create a routing agent with access to both tools.
agent = AssistantAgent(
    name="RoutingAgent",
    system_message=(
        "You are a routing agent that receives user queries and decides whether to use a text-to-SQL tool "
        "or a text-to-vector DB tool. If the query involves retrieving or manipulating data from a SQL database, "
    ),
    model_client=model_client,
    tools=[text_sqdb, text_vcdb]
)

async def main():
    cancellation_token = CancellationToken()
    
    # Example query for text-to-SQL routing.
    query_sql = "Retrieve the names of all customers from the database."
    response_sql = await agent.on_messages(
        [TextMessage(content=query_sql, source="user")],
        cancellation_token
    )
    print("SQL Routing Response:", response_sql.chat_message.content)
    
    # Example query for text-to-vector DB routing.
    query_vector = "Find similar documents to 'machine learning applications'."
    response_vector = await agent.on_messages(
        [TextMessage(content=query_vector, source="user")],
        cancellation_token
    )
    print("Vector DB Routing Response:", response_vector.chat_message.content)

if __name__ == "__main__":
    asyncio.run(main())
