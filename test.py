import agentguard
import openai
import os
from dotenv import load_dotenv


load_dotenv()

agentguard.init(
        public_key=os.getenv("AGENTGUARD_PUBLIC_KEY"),
        secret_key=os.getenv("AGENTGUARD_SECRET_KEY"),
        base_url=os.getenv("AGENTGUARD_BASE_URL"),
        project_id=os.getenv("AGENTGUARD_PROJECT_ID"),
    )

client = openai.OpenAI()

resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "email: sample@fmail.com"}],
)
print(resp.choices[0].message.content)
