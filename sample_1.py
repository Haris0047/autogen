import asyncio
import os
from dotenv import load_dotenv
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.agents import AssistantAgent
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_APIKEY")

model_client = OpenAIChatCompletionClient(
    model="gpt-4o",
    api_key=OPENAI_API_KEY,
    seed=42,
    temperature=0
)

class ParentAgent(AssistantAgent):
    async def on_messages(self, messages, cancellation_token):
        user_query = messages[-1].content

        clarity_response = await self.assess_clarity(user_query)
        if clarity_response.lower() != "proceed":
            return TextMessage(content="Please reenter your query with more clarity.", source=self.name)

        complexity = await self.evaluate_complexity(user_query)

        if complexity == "simple":
            response = await self.handle_simple_query(user_query)
        else:
            response = await self.handle_complex_query(user_query)

        return TextMessage(content=response, source=self.name)

    async def assess_clarity(self, query):
        return "Proceed"

    async def evaluate_complexity(self, query):
        return "simple"

    async def handle_simple_query(self, query):
        return f"Handled simple query: {query}"

    async def handle_complex_query(self, query):
        sub_queries = self.decompose_query(query)
        responses = [await self.handle_simple_query(sub_query) for sub_query in sub_queries]
        return " ".join(responses)

    def decompose_query(self, query):
        return [query]  

parent_agent = ParentAgent(
    name="parent_agent",
    system_message="You are a parent agent responsible for handling user queries through clarity assessment, complexity evaluation, decomposition, and RAG with fusion techniques. --If user query is generic answer it with greeting and offer a help.",
    model_client=model_client
)

async def main():
    cancellation_token = CancellationToken()
    user_query = "hi" 

    response = await parent_agent.on_messages(
        [TextMessage(content=user_query, source="user")],
        cancellation_token
    )
    print("Response:", response.content)

if __name__ == "__main__":
    asyncio.run(main())
