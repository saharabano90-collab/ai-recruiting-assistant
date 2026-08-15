import io
import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pydantic_core

# Document Parser Packages
from pypdf import PdfReader
from docx import Document
from PIL import Image

# Modern Gemini SDK Package 
from google import genai
from google.genai import types

app = FastAPI()

# Enable CORS for direct integration pipelines
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini Client (Ensure GEMINI_API_KEY is configured in your Vercel Project Dashboard)
client = genai.Client()

# --- Pydantic Layout Schemas ---
class CandidateProfile(BaseModel):
    name: str = Field(..., description="Extract candidate full name.")
    skills: List[str] = Field(default_factory=list, description="List of technical/soft skills.")
    experience_years: float = Field(0.0, description="Total summary of experience in years.")
    education: List[str] = Field(default_factory=list, description="Key degrees or universities.")
    certifications: List[str] = Field(default_factory=list, description="Professional certifications.")

class CandidateEvaluationResult(BaseModel):
    candidate_profile: CandidateProfile
    match_score: int = Field(..., description="Objective match score between 0 and 100.")
    summary: str = Field(..., description="Overview summarizing competency matches.")
    strengths: List[str] = Field(default_factory=list, description="Bullet items detailing strengths.")
    gaps: List[str] = Field(default_factory=list, description="Qualifications or technical elements missing.")
    interview_questions: List[str] = Field(default_factory=list, description="3-5 tailored technical/behavioral interview questions.")


# --- Document Parsing Utility Loops ---
def read_text_from_file_bytes(file_bytes: bytes, filename: str) -> str:
    ext = filename.lower().split('.')[-1] if '.' in filename else ''
    try:
        if ext == 'pdf':
            reader = PdfReader(io.BytesIO(file_bytes))
            return "".join([page.extract_text() or "" for page in reader.pages])
        elif ext in ['docx', 'doc']:
            doc = Document(io.BytesIO(file_bytes))
            return "\n".join([p.text for p in doc.paragraphs])
        elif ext in ['png', 'jpg', 'jpeg', 'webp']:
            image = Image.open(io.BytesIO(file_bytes))
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[image, "Transcribe all readable text formatting from this resume exactly."]
            )
            return response.text or ""
        else:
            raise HTTPException(status_code=400, detail="Unsupported format. Upload PDF, DOCX, PNG, or JPG.")
    except Exception as e:
        if isinstance(e, HTTPException): raise e
        raise HTTPException(status_code=422, detail=f"Failed to read file asset text parameters: {str(e)}")


# --- Serverless Function Endpoints ---

@app.post("/api/evaluate-direct-file")
async def evaluate_direct_file(
    job_requirements: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Serverless Endpoint: Accepts a raw job description string and a file asset attachment.
    Parses document bytes, triggers Gemini, and outputs a complete assessment block instantly.
    """
    file_bytes = await file.read()
    resume_text = read_text_from_file_bytes(file_bytes, file.filename or "resume.pdf")
    
    if not resume_text.strip():
        raise HTTPException(status_code=400, detail="Could not pull text data elements out of document.")

    ai_prompt = f"""
    You are HireAI's recruiting evaluation engine. 
    Analyze the provided candidate resume content and grade alignment against target requirements.
    
    [CORE REQUIREMENTS]
    {job_requirements}

    [CANDIDATE RESUME TEXT]
    {resume_text}

    Structure values objectively and format parameters exactly according to the schema template rules.
    """

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=ai_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=CandidateEvaluationResult,
                temperature=0.15,
            )
        )
        return CandidateEvaluationResult.model_validate_json(response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI engine grading fault: {str(e)}")
