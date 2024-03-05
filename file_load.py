from langchain.document_loaders import DirectoryLoader


def load_documents(directory="book"):
    loader = DirectoryLoader(directory, show_progress=True)
    documents = loader.load()
    for document in documents:
        print(document)


load_documents('book')