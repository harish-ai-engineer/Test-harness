# import os
# import agentguard
# import openai
# from dotenv import load_dotenv
# load_dotenv(override=True)

# os.environ["AGENTGUARD_CAPTURE_CONTENT"] = "true"

# agentguard.init(
#     public_key=os.getenv("AGENTGUARD_PUBLIC_KEY"),
#     secret_key=os.getenv("AGENTGUARD_SECRET_KEY"),
#     base_url=os.getenv("AGENTGUARD_BASE_URL"),
#     project_id=os.getenv("AGENTGUARD_PROJECT_ID"),
#     guardrails="auto",
# )

# client = openai.OpenAI(api_key= os.getenv("OPENAI_API_KEY"))
 
 
# with agentguard.policy(
#       user_id="test user",
#       session_id="test session",
#       feature="classify-intent",
#       metadata={"tenant": "acme-corp"},
#   ):
#       resp = client.chat.completions.create(
#           model="gpt-4o-mini",
#           messages=[{"role": "user", "content": "I want to buy this product now."}],
#       )
#       print(resp.choices[0].message.content)


