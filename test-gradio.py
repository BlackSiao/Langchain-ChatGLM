# 引入gradio为本地知识库做出一个精美的可交互web端
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import gradio as gr

# 加载并初始化模型
model = ChatOpenAI(temperature=1.0, model="gpt-3.5-turbo")

# 定义一个函数用来作为gr.interfact()的fn， 先不考虑history
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
