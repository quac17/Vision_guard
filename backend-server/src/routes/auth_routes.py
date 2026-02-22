from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from src.database import get_db
from src.models.models import User
from src.services.auth import verify_password, create_access_token, get_current_user
from src import schemas
from datetime import timedelta
import os

router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/login", response_model=schemas.Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 1440)))
    access_token = create_access_token(
        data={"sub": user.username, "role": user.role}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/me", response_model=schemas.UserResponse)
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user
