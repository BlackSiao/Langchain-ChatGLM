# 在这个示例中，将了解retrieval机制到底是用来干什么的，它是如何用来联系上下文的。
# 引入gradio为本地知识库做出一个精美的可交互web端
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import gradio as gr
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from langchain_community.vectorstores import Chroma


# 文件加载，直接加载本地book文件夹下的所有文件，并使用拆分器将其拆分
def load_documents(directory='book'):
    loader = DirectoryLoader(directory, show_progress=True, use_multithreading=True)
    documents = loader.load()
    # 加载文档后，要使得内容变得更加易于llm加载，就必须把长文本切割成一个个的小文本
    # tiktoken是OpenAI提供的 要考虑自己API里面就那么点钱别全霍霍了
    text_spliter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=100, chunk_overlap=0
    )
    split_docs = text_spliter.split_documents(documents)
    return split_docs

# 这次使用openai提供的embedding模型
def load_embedding():
    client = OpenAI()
    client.embeddings.create(
        model="text-embedding-ada-002",
        # input 应该设成load_documents返回的split_docs
        input="The food was delicious and the waiter...",
        encoding_format="float"
    )
    return

# 存储向量库
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

# 加载并初始化模型
model = ChatOpenAI(temperature=1.0, model="gpt-3.5-turbo")

# 定义一个函数用来作为gr.interfact()的fn，
def predict(message,history):
    template = """你是一名博学多才的图书馆管理员，对世界范围内的文学著作都如数家珍，
    你可以根据{Question}来准确的告诉读者，这本书的简介。对于你不了解的，你也会
    诚实的告诉问答者你不知道，而不是胡编乱造
    """
    # 提示词模板，其作用是输入问题给llm处理
    prompt = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()

    chain = prompt | model | output_parser
    response = chain.invoke({"Question": message})
    return response


# 这里我理解为可以将web交互端的输入作为predict函数的message，并返回对应的回答
demo = gr.ChatInterface(fn=predict,
                        examples=["今天天气如何？", "区块链是什么？", "我喜欢一个女孩子，该如何追求她"],
                        title="本地知识库问答系统")
if __name__ == "__main__":
    demo.launch()
