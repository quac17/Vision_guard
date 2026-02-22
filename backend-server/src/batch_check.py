import time
from sqlalchemy.orm import Session
from src.database import SessionLocal
from src.models.models import Attendance, Student
from src.services.email_service import send_attendance_email

def check_unsent_attendance():
    db: Session = SessionLocal()
    try:
        # Get only records with notified = 0 (unsent)
        unsent = db.query(Attendance).filter(Attendance.notified == 0).all()
        
        for record in unsent:
            student = db.query(Student).filter(Student.id == record.student_id).first()
            if student and student.email:
                print(f"Sending email for student: {student.full_name} to {student.email}")
                success = send_attendance_email(
                    to_email=student.email,
                    student_name=student.full_name,
                    status=record.status,
                    time=record.attendance_time.strftime("%Y-%m-%d %H:%M:%S")
                )
                
                if success:
                    record.notified = 2 # 2: Hoàn thành
                    print(f"Successfully notified parent of {student.full_name}")
                else:
                    record.notified = 1 # 1: Thất bại
                    print(f"Failed to send email for {student.full_name}")
                    
                db.commit()
            else:
                print(f"No student/email found for attendance record {record.id}")
                # Skip or mark as 1? Let's mark as 1 since it can't be sent.
                record.notified = 1
                db.commit()

    finally:
        db.close()

if __name__ == "__main__":
    print("Starting Vision Guard Attendance Check Batch...")
    while True:
        try:
            check_unsent_attendance()
        except Exception as e:
            print(f"Batch Error: {e}")
        
        # Check every 30 seconds
        time.sleep(30)
