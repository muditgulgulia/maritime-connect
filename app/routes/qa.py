from fastapi import APIRouter, HTTPException, Form, Depends
from sqlalchemy.orm import Session
from app.langchain.qa_chain import get_qa_chain
from app.db.database import get_db
from app.models.chat import ChatSession, ChatMessage
from uuid import UUID
from langchain_core.messages import HumanMessage, AIMessage
from sqlalchemy import desc
import json

router = APIRouter()

@router.post("/chat/new")
async def create_chat(question: str = Form(...), db: Session = Depends(get_db)):
    chat_session = ChatSession(user_id=1, title=question)
    db.add(chat_session)
    db.commit()
    db.refresh(chat_session)
    return {"session_id": chat_session.id, "title": chat_session.title}

@router.get("/chats")
async def get_chats(db: Session = Depends(get_db)):
    sessions = db.query(ChatSession).all()
    
    return {"sessions": sessions}


@router.post("/chat/{session_id}/ask")
async def ask_question(
    session_id: str,
    question: str = Form(...),
    db: Session = Depends(get_db)
):
    # Validate UUID format (optional but cleaner)
    try:
        uuid_obj = UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session ID format")

    # Get chat session and history
    session = db.query(ChatSession).filter(ChatSession.id == str(uuid_obj)).first()
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    
    # Get chat history
    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(desc(ChatMessage.created_at))
        .limit(10)
        .all()
    )
    
    chat_history = []

    for msg in messages:
        if msg.role == "user":
            chat_history.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            chat_history.append(AIMessage(content=msg.content))

    db.add(ChatMessage(session_id=session_id, role="user", content=question))
    db.commit()

    try:
        chain = await get_qa_chain()
        result = await chain.ainvoke({
            "question": question,
            "history": chat_history
        })
    except Exception as e:
        return {"error": "Invalid response format from AI", "raw_output": str(e)}


    # Store question and answer
    summary = result.get("summary", "")
    advice_points = result.get("advice_points", [])
    followup_questions = result.get("followup_questions", [])
    
    chat_msg = ChatMessage(
        session_id=session_id,
        role="assistant",
        content=summary,
        advice_points=json.dumps(advice_points) if advice_points else None,
        followup_questions=json.dumps(followup_questions) if followup_questions else None
    )

    db.add(chat_msg)
    db.commit()
    db.refresh(chat_msg)
    
    return {
        "session_id": session_id,
        "question": question,
        "answer": result
    }

@router.get("/chat/{session_id}/history")
async def get_chat_history(session_id: str, db: Session = Depends(get_db)):
    messages = db.query(ChatMessage).filter(
        ChatMessage.session_id == session_id
    ).order_by(ChatMessage.id).all()

    def try_parse_json(value):
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return None
        return value  # already a list or None

    history = [
        {
            "id": msg.id,
            "role": msg.role,
            "content": msg.content,
            "advice_points": try_parse_json(msg.advice_points),
            "followup_questions": try_parse_json(msg.followup_questions),
            "timestamp": msg.created_at,
        }
        for msg in messages
    ]

    return {"messages": history}
