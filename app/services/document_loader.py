from langchain_community.document_loaders import PyMuPDFLoader, TextLoader

class DocumentLoader:
    """
    Loads and parses documents from a file path.
    """

    def load(self, file_path):
        """
        Load documents from the given file path.
        """

        if file_path.endswith('.pdf'):
            loader = PyMuPDFLoader(file_path)
        else:
            loader = TextLoader(file_path)

        return loader.lazy_load()
