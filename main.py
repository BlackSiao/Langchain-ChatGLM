model_path = "D:\Hugface"

from langchain_community.llms import ChatGLM

endpoint_url ="http://127.0.0.1:8000"

llm = ChatGLM(
    endpoint_url =endpoint_url,
    max_token =80000,
    top_p=0.9
)

