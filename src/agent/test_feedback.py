from langsmith import traceable, Client as LangSmithClient
import os
from dotenv import load_dotenv

load_dotenv()


langsmith_client = LangSmithClient(api_key=os.getenv("LANGSMITH_API_KEY")) if os.getenv("LANGSMITH_API_KEY") else None

langsmith_client.create_feedback(
    run_id='019abc7b-ae04-7264-a923-f61395de8d81',
    key="user_feedback",
    score=1,
    comment="Great answer!"
)




