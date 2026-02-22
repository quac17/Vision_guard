# Database Schema Explanation

This document describes the database structure for the Vision Guard school bus attendance system.

## Entities

### 1. Users (`users`)
Stores both admin and parent accounts.
- `role`: Distinguishes between 'admin' (can create parents, view all students) and 'parent' (can register students and receive notifications).
- `hashed_password`: Stored securely using bcrypt.

### 2. Students (`students`)
Stores student details.
- `parent_id`: Link to the parent account that registered the student.
- `student_code`: Unique identifier for the student (e.g., RFID or QR code ID).

### 3. Student Photos (`student_photos`)
Stores paths to student photos for facial recognition/identification.
- A student can have 3 to 5 photos as required.

### 4. Attendance (`attendance`)
Logs attendance events from edge devices.
- `status`: 'on_bus' or 'off_bus'.
- `attendance_time`: The actual time the event occurred.
- `notified`: Tracking flag for the batch email service.

## Relationships
- A `User` (parent) can have multiple `Students`.
- A `Student` has multiple `Student Photos`.
- A `Student` has multiple `Attendance` logs.
