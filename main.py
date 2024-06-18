from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
from predict import extract_id

app = FastAPI()

class Employee(BaseModel):
    name: str
    age: int
    position: str
    remote: bool
    filename: str
    content_type: str

@app.get("/")
async def root():
    return HTMLResponse('<h1>Worked Successfully</h1>')

@app.post("/image", response_model=str)
async def upload_image(
    name: str = Form(...),
    age: int = Form(...),
    position: str = Form(...),
    remote: bool = Form(...),
    image: UploadFile = File(...)
):
    return extract_id(image.filename)

#if __name__ == "__main__":
#    import uvicorn
#    uvicorn.run(app, host="0.0.0.0", port=8000)
