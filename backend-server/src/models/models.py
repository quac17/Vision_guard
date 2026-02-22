from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Boolean, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from src.database import Base
import enum

class UserRole(str, enum.Enum):
    admin = "admin"
    parent = "parent"

class AttendanceStatus(str, enum.Enum):
    on_bus = "on_bus"
    off_bus = "off_bus"

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, nullable=False)
    email = Column(String)
    phone = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    students = relationship("Student", back_populates="parent")

class Student(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String, nullable=False)
    grade = Column(String, nullable=False)
    student_code = Column(String, unique=True, index=True, nullable=False)
    parent_id = Column(Integer, ForeignKey("users.id"))
    email = Column(String)
    phone = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    parent = relationship("User", back_populates="students")
    photos = relationship("StudentPhoto", back_populates="student")
    attendance_logs = relationship("Attendance", back_populates="student")

class StudentPhoto(Base):
    __tablename__ = "student_photos"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.id"))
    photo_path = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    student = relationship("Student", back_populates="photos")

class Attendance(Base):
    __tablename__ = "attendance"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.id"))
    status = Column(String, nullable=False)
    attendance_time = Column(DateTime(timezone=True), server_default=func.now())
    notified = Column(Integer, default=0) # 0: chưa gửi, 1: gửi thất bại, 2: gửi thành công
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    student = relationship("Student", back_populates="attendance_logs")
