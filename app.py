import os
from langchain_openai.embeddings import OpenAIEmbeddings

# Set your API key
OPENAI_API_KEY = "sk-8cd1123a9bd04d8fbe0b1f985e4bdc75"

# Initialize embeddings client with DashScope
embeddings = OpenAIEmbeddings(
    model="text-embedding-v3",
    openai_api_key=OPENAI_API_KEY,
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)

# ✅ Direct call to the underlying client (same as your working OpenAI code)
resp = embeddings.client.create(
    model="text-embedding-v3",
    input="Who is the Prime Minister of India?",
    encoding_format="float"
)

embedding_vector = resp.data[0].embedding
print("Embedding length:", len(embedding_vector))
print("First 10 values:", embedding_vector[:10])
