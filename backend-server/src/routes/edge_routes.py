from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from src.database import get_db
from src.models.models import Student, Attendance
from src import schemas
from datetime import datetime

router = APIRouter(prefix="/edge", tags=["edge"])

@router.post("/attendance", response_model=schemas.AttendanceResponse)
async def submit_attendance(
    data: schemas.AttendanceCreate,
    db: Session = Depends(get_db)
):
    # Find student by code
    student = db.query(Student).filter(Student.student_code == data.student_code).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    
    # Create attendance record
    new_attendance = Attendance(
        student_id=student.id,
        status=data.status,
        attendance_time=data.attendance_time
    )
    db.add(new_attendance)
    db.commit()
    db.refresh(new_attendance)
    return new_attendance
