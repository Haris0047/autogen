import asyncio
import nest_asyncio
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.agents import AssistantAgent
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os
from dotenv import load_dotenv

nest_asyncio.apply()  # Allow nested event loops

async def main() -> None:
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    model_client = OpenAIChatCompletionClient(model="gpt-4o", api_key=OPENAI_API_KEY, seed=42, temperature=0)

    assistant = AssistantAgent(
        name="assistant",
        system_message="You are a helpful assistant.",
        model_client=model_client,
    )

    cancellation_token = CancellationToken()
    response = await assistant.on_messages(
        [TextMessage(content="Hello!", source="user")],
        cancellation_token
    )
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
