from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

stream = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "你能做什么"}],
    stream=True,
    timeout=120  # Set timeout to 120 seconds
)

for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")