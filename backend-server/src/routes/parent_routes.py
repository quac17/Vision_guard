from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form
from sqlalchemy.orm import Session
from typing import List
import os
import uuid
from src.database import get_db
from src.models.models import User, Student, StudentPhoto
from src.services.auth import get_current_parent
from src import schemas

router = APIRouter(prefix="/parent", tags=["parent"])

UPLOAD_DIR = "uploads/students"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/register-student", response_model=schemas.StudentResponse)
async def register_student(
    full_name: str = Form(...),
    grade: str = Form(...),
    student_code: str = Form(...),
    email: str = Form(None),
    phone: str = Form(None),
    photos: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
    parent: User = Depends(get_current_parent)
):
    # Validate photo count
    if not (3 <= len(photos) <= 5):
        raise HTTPException(status_code=400, detail="Please upload 3 to 5 photos")

    # Check student code
    db_student = db.query(Student).filter(Student.student_code == student_code).first()
    if db_student:
        raise HTTPException(status_code=400, detail="Student code already exists")

    # Create Student
    new_student = Student(
        full_name=full_name,
        grade=grade,
        student_code=student_code,
        parent_id=parent.id,
        email=email or parent.email,
        phone=phone or parent.phone
    )
    db.add(new_student)
    db.commit()
    db.refresh(new_student)

    # Save Photos
    for photo in photos:
        file_ext = photo.filename.split('.')[-1]
        file_name = f"{new_student.id}_{uuid.uuid4()}.{file_ext}"
        file_path = os.path.join(UPLOAD_DIR, file_name)
        
        with open(file_path, "wb") as buffer:
            buffer.write(await photo.read())
        
        new_photo = StudentPhoto(student_id=new_student.id, photo_path=file_path)
        db.add(new_photo)
    
    db.commit()
    return new_student
