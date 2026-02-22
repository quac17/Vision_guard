from pydantic import BaseModel, EmailStr
from typing import List, Optional
from datetime import datetime
from .models.models import UserRole, AttendanceStatus

# User Schemas
class UserBase(BaseModel):
    username: str
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    role: UserRole

class UserCreate(UserBase):
    password: str

class UserResponse(UserBase):
    id: int
    created_at: datetime
    class Config:
        from_attributes = True

# Token Schemas
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[str] = None

# Student Schemas
class StudentBase(BaseModel):
    full_name: str
    grade: str
    student_code: str
    email: Optional[EmailStr] = None
    phone: Optional[str] = None

class StudentCreate(StudentBase):
    parent_id: Optional[int] = None

class StudentResponse(StudentBase):
    id: int
    parent_id: Optional[int]
    created_at: datetime
    class Config:
        from_attributes = True

# Attendance Schemas
class AttendanceCreate(BaseModel):
    student_code: str
    status: AttendanceStatus
    attendance_time: datetime = datetime.now()

class AttendanceResponse(BaseModel):
    id: int
    student_id: int
    status: str
    attendance_time: datetime
    notified: int
    class Config:
        from_attributes = True
