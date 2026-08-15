import os
import io
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pydantic_core

# Document & Image Reading Libraries
from pypdf import PdfReader
from docx import Document
from PIL import Image

# Modern Gemini SDK
from google import genai
from google.genai import types

app = FastAPI(title="AI Recruiting Assistant API")

# Enable CORS for frontend connectivity
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the Gemini Client (Ensure GEMINI_API_KEY environment variable is set)
client = genai.Client()

# --- Pydantic Data Structures ---
class CandidateProfile(BaseModel):
    name: str
    skills: List[str]
    experience_years: float
    education: List[str]
    certifications: List[str]

class MatchEvaluation(BaseModel):
    candidate_profile: CandidateProfile
    match_score: int
    summary: str
    strengths: List[str]
    gaps: List[str]
    generated_interview_questions: List[str]


# --- Document Extraction Functions ---

def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse PDF file: {str(e)}")

def extract_text_from_docx(file_bytes: bytes) -> str:
    try:
        doc = Document(io.BytesIO(file_bytes))
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse Word Document: {str(e)}")

def extract_text_from_image(file_bytes: bytes) -> str:
    """
    Passes image bytes (PNG/JPG) directly into Gemini's multimodal 
    engine to read the text content natively.
    """
    try:
        image = Image.open(io.BytesIO(file_bytes))
        ocr_prompt = "Extract and transcribe all readable text from this resume document accurately as raw text."
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[image, ocr_prompt]
        )
        return response.text or ""
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse or run OCR on Image: {str(e)}")


# --- Core AI Processing Logic ---

def process_ai_evaluation(job_description: str, resume_text: str) -> MatchEvaluation:
    if not job_description.strip() or not resume_text.strip():
        raise HTTPException(status_code=400, detail="Job description or resume content cannot be blank.")

    prompt = f"""
    You are an expert HR AI Recruiting Assistant. Your task is to extract candidate details from the provided resume text, 
    evaluate how well they match the job description, score them, and generate role-specific interview questions.

    [JOB DESCRIPTION]
    {job_description}

    [CANDIDATE RESUME]
    {resume_text}

    Analyze the profiles objectively and follow the exact required JSON schema structure.
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=MatchEvaluation,
                temperature=0.2,
            ),
        )
        return MatchEvaluation.model_validate_json(response.text)
    except pydantic_core.ValidationError as ve:
        raise HTTPException(status_code=500, detail=f"AI data did not match HR schema: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Service communication failure: {str(e)}")


# --- API Endpoints ---

@app.post("/api/evaluate-candidate", response_model=MatchEvaluation)
async def evaluate_candidate_text(payload: dict):
    """Fallback text submission endpoint"""
    return process_ai_evaluation(payload.get("job_description", ""), payload.get("resume_text", ""))


@app.post("/api/evaluate-file", response_model=MatchEvaluation)
async def evaluate_candidate_file(
    job_description: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Main file submission endpoint. Supports .pdf, .docx, and common image formats.
    """
    file_bytes = await file.read()
    filename = file.filename.lower() if file.filename else ""
    resume_text = ""

    if filename.endswith('.pdf'):
        resume_text = extract_text_from_pdf(file_bytes)
    elif filename.endswith('.docx') or filename.endswith('.doc'):
        resume_text = extract_text_from_docx(file_bytes)
    elif filename.endswith(('.png', '.jpg', '.jpeg', '.webp')):
        resume_text = extract_text_from_image(file_bytes)
    else:
        raise HTTPException(
            status_code=400, 
            detail="Unsupported file extension. Please upload a PDF, Word document, or Image (PNG/JPG)."
        )

    if not resume_text.strip():
        raise HTTPException(status_code=400, detail="Could not extract any readable text from the uploaded document.")

    return process_ai_evaluation(job_description, resume_text)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
