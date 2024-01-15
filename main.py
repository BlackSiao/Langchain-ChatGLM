from transformers import AutoTokenizer, AutoModel
# 已修改模型位置为下载后的位置，确认模型完整无误
tokenizer = AutoTokenizer.from_pretrained("D:\Hugface", trust_remote_code=True)
model = AutoModel.from_pretrained("D:\Hugface", trust_remote_code=True).half().cuda()
model = model.eval()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)
response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
print(response)