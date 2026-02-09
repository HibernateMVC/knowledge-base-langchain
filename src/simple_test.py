import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.responses import FileResponse
import uvicorn

app = FastAPI()

@app.get("/")
async def root():
    print("Root endpoint called")
    frontend_path = os.path.join("frontend", "index.html")
    print(f"Looking for file at: {frontend_path}")
    if os.path.exists(frontend_path):
        print("File exists, returning FileResponse")
        return FileResponse(frontend_path)
    else:
        print("File does not exist, returning dict")
        return {"message": "Frontend file not found"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)