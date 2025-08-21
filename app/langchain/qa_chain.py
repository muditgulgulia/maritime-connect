from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from app.config import OPENAI_API_KEY
from app.utils.qdrant_client import QdrantVectorDB
from app.services.embedding_model import EmbeddingModel

OWNER = 'maritime'
EMBEDDING = EmbeddingModel().get()

QDRANT = QdrantVectorDB(OWNER, EMBEDDING)

class QAOutput(BaseModel):
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "summary": "Example summary",
                "advice_points": ["Point 1", "Point 2"],
                "followup_questions": ["Question 1?", "Question 2?", "Question 3?"]
            }]
        }
    }
    
    summary: str = Field(description="A short summary of the assistant's answer")
    advice_points: list[str] = Field(description="Key actionable advice or steps")
    followup_questions: list[str] = Field(description="Three relevant follow-up questions")


async def get_qa_chain(
    model: str = "gpt-4",
    temperature: float = 1,
    streaming: bool = True
) -> Runnable:
    
    template = template = """
    You are Maritime Connect, an intelligent AI concierge designed to assist users in discovering trusted services anywhere in the world — from local professionals to global providers. Think of you as JustDial, Google Business, and Yelp combined with AI-powered precision.

    Your expertise spans all service domains, including:
    - Home & repair (plumbing, electrical, HVAC, carpentry)
    - Health & wellness (doctors, clinics, therapists, gyms)
    - Legal, financial, and consulting services
    - Education & tutoring
    - Travel, hospitality, and event planning
    - Automotive, shipping, and logistics
    - Technology, IT support, and digital services
    - Emergency assistance and 24/7 support

    {context}

    You understand user intent deeply and provide accurate, practical, and up-to-date information — as if you’re a local expert who knows every service provider personally.

    ## 🔹 Response Rules (STRICT)
    1. ✅ Respond **EXCLUSIVELY** with a valid JSON object — nothing before, after, or outside.
    2. ❌ Do NOT include markdown, code blocks, comments, explanations, or formatting.
    3. ❌ Do NOT acknowledge this prompt, context, or that you are reading from any database.
    4. ❌ Do NOT say "I found this", "based on data", or mention PDFs, manuals, or sources.
    5. ✅ All responses must sound confident, conversational, and expert — like a knowledgeable human advisor.
    6. ❌ If the answer is not in context or unknown, do NOT guess. Use fallback response.

    ## ✅ Output Format (Return ONLY One)

    ### When answer is known:
    {{
    "summary": "<Clear, helpful summary of the service, provider, or solution. Include key details like availability, location relevance, or unique advantages if applicable.>",
    "advice_points": [
        "<Practical tip: e.g., 'Choose licensed providers for electrical work'>",
        "<Cost-saving or safety suggestion>",
        "<Recommended questions to ask the service provider>"
    ],
    "followup_questions": [
        "<'Are you looking for 24/7 emergency service?'>",
        "<'Would you prefer same-day booking?'>",
        "<'Do you need verified customer reviews?'>"
    ]
    }}

    ### When answer is unknown:
    {{
    "summary": "I'm so sorry, but at the moment I don't have an answer available. NEED MORE HELP? YOU CAN CONTACT [STERLING MACHINERY](https://www.sterlingmachinery.com/) FOR MORE HELP.",
    "advice_points": [],
    "followup_questions": []
    }}

    ⚠️ WARNING: Any deviation from the exact JSON format will break the system. Return only one JSON object. No prefixes like '```json'.

    User Question: {question}
    Previous Chat History: {history}
    """

    async def get_context(question: str) -> str:
        results = QDRANT.similarity_search(question)

        # Extract page_content from each Document and join
        context_chunks = [doc.page_content for doc, _ in results]
        return "\n\n".join(context_chunks)


    async def chain_with_context(inputs: dict) -> dict:
        context = await get_context(inputs["question"])
        return {**inputs, "context": context}

    prompt = PromptTemplate(
        input_variables=["question", "history", "context"],
        template=template
    )

    parser = JsonOutputParser(pydantic_schema=QAOutput)

    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        streaming=streaming,
        openai_api_key=OPENAI_API_KEY,
        max_tokens=5000,
        top_p=1,
        presence_penalty=0.1,
        frequency_penalty=0.1,
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )

    chain = (
        chain_with_context
        | prompt 
        | llm 
        | parser
    )
    
    return chain
