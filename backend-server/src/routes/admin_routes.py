from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from src.database import get_db
from src.models.models import User, Student, UserRole
from src.services.auth import get_current_admin, get_password_hash
from src import schemas

router = APIRouter(prefix="/admin", tags=["admin"])

# Account Management for Parents
@router.post("/parents", response_model=schemas.UserResponse)
async def create_parent_account(
    parent_data: schemas.UserCreate, 
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin)
):
    # Check if parent exists
    db_user = db.query(User).filter(User.username == parent_data.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_pwd = get_password_hash(parent_data.password)
    new_parent = User(
        username=parent_data.username,
        hashed_password=hashed_pwd,
        role=UserRole.parent,
        email=parent_data.email,
        phone=parent_data.phone
    )
    db.add(new_parent)
    db.commit()
    db.refresh(new_parent)
    return new_parent

# Student Management (List, Add, Update, Delete)
@router.get("/students", response_model=List[schemas.StudentResponse])
async def list_students(db: Session = Depends(get_db), admin: User = Depends(get_current_admin)):
    return db.query(Student).all()

@router.post("/students", response_model=schemas.StudentResponse)
async def add_student(
    student_data: schemas.StudentCreate,
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin)
):
    db_student = db.query(Student).filter(Student.student_code == student_data.student_code).first()
    if db_student:
        raise HTTPException(status_code=400, detail="Student code already exists")
    
    new_student = Student(**student_data.dict())
    db.add(new_student)
    db.commit()
    db.refresh(new_student)
    return new_student

@router.put("/students/{student_id}", response_model=schemas.StudentResponse)
async def update_student(
    student_id: int,
    student_data: schemas.StudentBase,
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin)
):
    db_student = db.query(Student).filter(Student.id == student_id).first()
    if not db_student:
        raise HTTPException(status_code=404, detail="Student not found")
    
    for key, value in student_data.dict().items():
        setattr(db_student, key, value)
    
    db.commit()
    db.refresh(db_student)
    return db_student

@router.delete("/students/{student_id}")
async def delete_student(
    student_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin)
):
    db_student = db.query(Student).filter(Student.id == student_id).first()
    if not db_student:
        raise HTTPException(status_code=404, detail="Student not found")
    
    db.delete(db_student)
    db.commit()
    return {"message": "Student deleted"}
