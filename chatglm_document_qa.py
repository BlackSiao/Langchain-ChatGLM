import time
import os
import gradio as gr
from langchain.document_loaders import DirectoryLoader
from langchain.llms import ChatGLM
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA


def load_documents(directory="book"):
    loader = DirectoryLoader(directory, show_progress=True)
    documents = loader.load()
    text_spliter = CharacterTextSplitter(chunk_size=256, chunk_overlap=0)
    docs_spliter = text_spliter.split_documents(documents)
    return docs_spliter


# # 加载embedding
# embedding_model_dict = {
#     "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
#     "ernie-base": "nghuyong/ernie-3.0-base-zh",
#     "text2vec": "GanymedeNil/text2vec-large-chinese",
#     "text2vec2": "uer/sbert-base-chinese-nli",
#     "text2vec3": "shibing624/text2vec-base-chinese",
# }

def load_embedding_model(model_name="ernie-tiny"):
    """
    加载embedding模型
    :param model_name:
    :return:
    """
    encode_kwargs = {"normalize_embeddings": False}
    model_kwargs = {"device": "cuda:0"}
    return HuggingFaceEmbeddings(
        model_name=r"C:\Users\BlackSiao\Desktop\毕业设计\text2vec",  #手动下载模型到本地'text2vec文件夹'
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )


def store_chroma(docs, embeddings, persist_directory="VectorStore"):
    """
    讲文档向量化，存入向量数据库
    :param docs:
    :param embeddings:
    :param persist_directory:
    :return:
    """
    db = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
    db.persist()
    return db


# 这段代码不是很明白
embedding = load_embedding_model('text2vec3')
if not os.path.exists('VectorStore'):
    documents = load_documents()
    db = store_chroma(documents, embedding)
else:
    db = Chroma(persist_directory='VectorStore', embedding_function=embedding)

# 创建llm
llm = ChatGLM(
    endpoint_url='http://127.0.0.1:8000',
    max_token=80000,
    top_p=0.9
)
# 创建qa
QA_CHAIN_PROMPT = PromptTemplate.from_template("""根据下面的上下文（context）内容回答问题。
如果你不知道答案，就回答不知道，不要试图编造答案。
答案最多3句话，保持答案简介。
总是在答案结束时说”谢谢你的提问！“
{context}
问题：{question}
""")
retriever = db.as_retriever()
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    verbose=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)


# def chat(question, history):
#     response = qa.run(question)
#     return response
#
#
# # 调用gradio生成本地的web交互端
# demo = gr.ChatInterface(chat)
# demo.launch(inbrowser=True)

def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)


def add_text(history, text):
    history = history + [(text, None)]
    return history, gr.Textbox(value="", interactive=False)


def add_file(history, file):
    directory = os.path.dirname(file.name)  # 拿到临时文件夹
    documents = load_documents(directory)
    store_chroma(documents, embedding)   #将临时上传的加载好，并存到数据库里面
    history = history + [((file.name,), None)]
    return history


def bot(history):
    message = history[-1][0]
    if isinstance(message, tuple):
        response = "文件上传成功！！"
    else:
        response = qa({"query": message})['result']  # 这里也进行了修改
    history[-1][1] = ""
    for character in response:
        history[-1][1] += character
        time.sleep(0.05)
        yield history


with gr.Blocks() as demo:
    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False,
        avatar_images=(None, (os.path.join(os.path.dirname(__file__), "avatar.png"))),
    )

    with gr.Row():
        txt = gr.Textbox(
            scale=4,
            show_label=False,
            placeholder="Enter text and press enter, or upload an image",
            container=False,
        )
        btn = gr.UploadButton("📁", file_types=['txt'])   # 限制上传文档类型为txt

    txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
        bot, chatbot, chatbot, api_name="bot_response"
    )
    txt_msg.then(lambda: gr.Textbox(interactive=True), None, [txt], queue=False)
    file_msg = btn.upload(add_file, [chatbot, btn], [chatbot], queue=False).then(
        bot, chatbot, chatbot
    )

    chatbot.like(print_like_dislike, None, None)


demo.queue()
if __name__ == "__main__":
    demo.launch()

