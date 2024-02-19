# model_path = "D:\Hugface"
#
# from langchain_community.llms import ChatGLM
#
# endpoint_url ="http://127.0.0.1:8000"
#
# llm = ChatGLM(
#     endpoint_url =endpoint_url,
#     max_token =80000,
#     top_p=0.9
# )

from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("D:\Hugface", trust_remote_code=True)
model = AutoModel.from_pretrained("D:\Hugface", trust_remote_code=True).half().cuda()
model = model.eval()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)
response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
print(response)
