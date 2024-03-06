from langchain.document_loaders import DirectoryLoader

# 验证是否能够完成上传本地文档功能

def load_documents(directory="book"):
    loader = DirectoryLoader(directory, show_progress=True)
    documents = loader.load()
    for document in documents:
        print(document)


load_documents('book')