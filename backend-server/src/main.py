from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.routes import auth_routes, admin_routes, parent_routes, edge_routes
import uvicorn
import os

app = FastAPI(title="Vision Guard API", description="Backend for School Bus Attendance Tracking")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(auth_routes.router)
app.include_router(admin_routes.router)
app.include_router(parent_routes.router)
app.include_router(edge_routes.router)

@app.get("/")
async def root():
    return {"message": "Welcome to Vision Guard API"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
