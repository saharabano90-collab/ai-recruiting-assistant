import os
import json
import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from pydantic import BaseModel, Field
from typing import List

app = FastAPI()

# Enable CORS so your frontend can communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenRouter Configuration (Fetched from environment variables)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("CRITICAL ERROR: OPENROUTER_API_KEY environment variable is not set.")

OPENROUTER_URL = "https://openrouter.ai"

# Define the exact JSON structure we want the model to return
class CandidateEvaluation(BaseModel):
    candidate_name: str = Field(description="Full name of the candidate")
    skills: List[str] = Field(description="Key skills found in the resume")
    experience_years: int = Field(description="Estimated total years of professional experience")
    match_score: int = Field(description="A score from 1 to 100 based on job fit")
    summary: str = Field(description="A 2-3 sentence concise profile summary")
    strengths: List[str] = Field(description="Key qualifications aligning with the job description")
    gaps: List[str] = Field(description="Required job skills or experience missing from the resume")
    interview_questions: List[str] = Field(description="3-4 targeted, role-specific questions to ask this candidate")

def extract_text_from_pdf(file: UploadFile) -> str:
    """Helper function to parse text out of an uploaded PDF file."""
    try:
        reader = PdfReader(file.file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {str(e)}")

@app.get("/api")
def read_root():
    return {"status": "Backend API is running successfully!"}

@app.get("/api/docs")
def custom_docs():
    return app.openapi()

@app.post("/api/evaluate", response_model=CandidateEvaluation)
async def evaluate_candidate(
    job_description: str = Form(...),
    resume_file: UploadFile = File(...)
):
    # 1. Parse the PDF
    resume_text = extract_text_from_pdf(resume_file)
    if not resume_text:
        raise HTTPException(status_code=400, detail="Could not extract text from the provided resume PDF.")

    # 2. Build the AI prompt with explicit JSON schema instructions for OpenRouter
    prompt = f"""
    You are an expert HR recruitment assistant. Analyze the candidate's Resume against the provided Job Description.
    
    JOB DESCRIPTION:
    {job_description}
    
    CANDIDATE RESUME:
    {resume_text}
    
    Extract details, compare qualifications, identify strengths/gaps, score the candidate, and generate interview questions.
    
    You MUST respond with a raw JSON object matching this schema:
    {json.dumps(CandidateEvaluation.model_json_schema(), indent=2)}
    """

    # 3. Call Google Gemini via OpenRouter API
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "google/gemini-2.5-flash",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"} 
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60.0)
            
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"OpenRouter API Error: {response.text}")
            
        result_json = response.json()
        raw_content = result_json["choices"]["message"]["content"]
        
        # 4. Parse and validate the stringified JSON against your Pydantic schema
        validated_data = CandidateEvaluation.model_validate_json(raw_content)
        return validated_data

    except KeyError:
        raise HTTPException(status_code=500, detail="Unexpected response structure from OpenRouter.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation Error: {str(e)}")
