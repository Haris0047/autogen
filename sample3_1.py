import os
import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken

# Initialize the OpenAI model client
model_client = OpenAIChatCompletionClient(
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY"),
    seed=42,
    temperature=0
)

# Define the child agent specialized for handling complex queries.
child_agent = AssistantAgent(
    name="ChildAgent",
    system_message=(
        "You are a child agent specialized in handling complex queries. "
        "Provide detailed analysis and comprehensive answers for queries that require multiple components or in-depth reasoning."
    ),
    model_client=model_client
)

# Define the ParentAgent which manages the overall workflow.
class ParentAgent(AssistantAgent):
    async def on_messages(self, messages, cancellation_token):
        user_query = messages[-1].content

        # Step 1: Assess clarity.
        clarity_response = await self.assess_clarity(user_query)
        if clarity_response.lower() != "proceed":
            return TextMessage(
                content="Please reenter your query with more clarity.", source=self.name
            )

        # Step 2: Evaluate complexity.
        complexity = await self.evaluate_complexity(user_query)

        # Step 3: Process the query based on its complexity.
        if complexity == "simple":
            response = await self.handle_simple_query(user_query)
        else:
            response = await self.delegate_to_child(user_query, cancellation_token)

        return TextMessage(content=response, source=self.name)

    async def assess_clarity(self, query: str):
        # In a production system, you might use additional prompts or logic.
        # For now, we assume the query is clear.
        return "Proceed"

    async def evaluate_complexity(self, query: str):
        # Simple heuristic: if the query contains "and" or more than one question mark, consider it complex.
        if " and " in query.lower() or query.count("?") > 1:
            return "complex"
        else:
            return "simple"

    async def handle_simple_query(self, query: str):
        # Simulate a RAG with fusion response for simple queries.
        return f"Simple RAG response: The answer to '{query}' is based on concise analysis."

    async def delegate_to_child(self, query: str, cancellation_token: CancellationToken):
        # Delegate the query to the child agent for complex queries.
        child_response = await child_agent.on_messages(
            [TextMessage(content=query, source="user")],
            cancellation_token
        )
        return child_response.chat_message.content

# Initialize the parent agent with its system message.
parent_agent = ParentAgent(
    name="ParentAgent",
    system_message=(
        "You are a parent agent responsible for processing user queries with the following workflow:\n"
        "1. Assess clarity: If the query is ambiguous or incomplete, ask the user to reenter it with more clarity.\n"
        "2. Evaluate complexity: Determine if the query is simple or complex.\n"
        "   - A simple query is a single, straightforward question.\n"
        "   - A complex query involves multiple components or requires detailed explanation.\n"
        "3. Process the query:\n"
        "   - For simple queries, perform a concise RAG with fusion analysis.\n"
        "   - For complex queries, delegate to a child agent for detailed processing.\n"
        "Ensure each step is executed thoroughly before returning the final answer."
    ),
    model_client=model_client
)

async def main():
    cancellation_token = CancellationToken()
    # Example query (contains "and", so it's treated as complex).
    while True:
        user_query = input("User: ")
        if user_query == 'exit':
            break
        # user_query = "What is Microsoft's annual EPS and revenue breakdown for 2023?"
    
        response = await parent_agent.on_messages(
            [TextMessage(content=user_query, source="user")],
            cancellation_token
        )
        print("Final Response:", response.content)

if __name__ == "__main__":
    asyncio.run(main())
