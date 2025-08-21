from fastapi import APIRouter, HTTPException, Form
from app.auth.jwt import create_access_token
from datetime import timedelta

router = APIRouter()

# Dummy user
FAKE_USER = {
    "username": "admin",
    "password": "12345678"  # You can use hash in production
}

@router.post("/login")
def login(username: str = Form(...), password: str = Form(...)):
    if username != FAKE_USER["username"] or password != FAKE_USER["password"]:
        raise HTTPException(status_code=400, detail="Incorrect credentials")
    access_token = create_access_token(
        data={"sub": username},
        expires_delta=timedelta(minutes=60)
    )
    return {"access_token": access_token, "token_type": "bearer"}
