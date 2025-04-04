import os
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
import asyncio
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken

model_client = OpenAIChatCompletionClient(
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY"),
    seed=42,
    temperature=0
)

parent_agent = AssistantAgent(
    name="parent_agent",
    system_message=(
        "You are a parent agent responsible for handling user queries through the following steps:\n"
        "1. **Assess Clarity:** Evaluate if the user's query is clear and complete. "
        "If ambiguous or incomplete, prompt the user to reenter the query with more clarity.\n"
        "2. **Evaluate Complexity:** Determine if the query is simple or complex. "
        "A simple query is a single, straightforward question, while a complex query involves multiple components or requires detailed explanation.\n"
        "3. **Process the Query:**\n"
        "   - **Simple Query:** Perform Retrieval-Augmented Generation (RAG) with fusion to retrieve relevant information and generate a concise response.\n"
        "   - **Complex Query:** Decompose the query into distinct sub-queries, perform RAG with fusion for each, and aggregate the information to generate a comprehensive response.\n"
        "Ensure that each step is executed thoroughly before proceeding to the next."
    ),
    model_client=model_client
)

async def main():
    user_query=""
    while True:
        if user_query == 'exit':
            break
        cancellation_token = CancellationToken()
        user_query = "What is Microsoft's annual EPS?"  

        response = await parent_agent.on_messages(
            [TextMessage(content=user_query, source="user")],
            cancellation_token
        )
        print("Response:", response.chat_message.content)

if __name__ == "__main__":
    asyncio.run(main())


