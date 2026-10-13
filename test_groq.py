
import os
from dotenv import load_dotenv
from groq import Groq

# Load the API key from .env
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env")

# Connect to Groq
client = Groq(api_key=api_key)

# Send a test request
response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {
            "role": "user",
            "content": "Reply with exactly: PaperLens connection successful!"
        }
    ],
    temperature=0,
)

print(response.choices[0].message.content)