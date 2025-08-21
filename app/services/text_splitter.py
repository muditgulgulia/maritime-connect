from langchain.text_splitter import RecursiveCharacterTextSplitter

class TextSplitter:
    """
    Splits documents into text chunks using configurable size and overlap.
    """

    def __init__(self, chunk_size=1000, chunk_overlap=100):
        """
        Initialize the splitter.
        """
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def split(self, docs):
        """
        Split documents into chunks.
        """
        return self.splitter.split_documents(docs)
