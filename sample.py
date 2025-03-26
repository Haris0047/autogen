import asyncio
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage

def get_weather(city: str) -> str: # Async tool is possible too.
    return f"The weather in {city} is 72 degree and sunny."

async def main() -> None:
    model_client = OpenAIChatCompletionClient(model="gpt-4o", seed=42, temperature=0)
    assistant = AssistantAgent(
        name="assistant",
        system_message="You are a helpful assistant. You can call tools to help user.",
        model_client=model_client,
        tools=[get_weather],
        reflect_on_tool_use=False, # Set to True to have the model reflect on the tool use, set to False to return the tool call result directly.
    )
    while True:
        user_input = input("User: ")
        if user_input == "exit":
            break
        response = await assistant.on_messages([TextMessage(content=user_input, source="user")], CancellationToken())
        print("Assistant:", response.chat_message.content)

asyncio.run(main())
