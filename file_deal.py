import os
import gradio as gr
from langchain.document_loaders import DirectoryLoader
from langchain.llms import ChatGLM
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

# 对文档进行切片，向量化，存储
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