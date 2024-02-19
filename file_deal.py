from langchain_community.document_loaders import DirectoryLoader


def load_documents(directory="book"):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    for document in documents:
        print(document)


load_documents()
