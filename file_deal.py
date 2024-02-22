from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter


def load_documents(directory="./book"):
    loader = DirectoryLoader(directory, show_progress=True)
    documents = loader.load()
    text_spliter = CharacterTextSplitter(chunk_size=256, chunk_overlap=0)
    docs_spliter = text_spliter.split_documents(documents)

    print(docs_spliter[:2])
    return docs_spliter


load_documents()

