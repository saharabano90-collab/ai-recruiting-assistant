import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from openai import OpenAI  # Swapped from google.genai
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

# Define the exact JSON structure we want OpenRouter to return
class CandidateEvaluation(BaseModel):
    candidate_name: str = Field(description="Full name of the candidate")
    skills: List[str] = Field(description="Key skills found in the resume")
    experience_years: int = Field(description="Estimated total years of professional experience")
    match_score: int = Field(description="A score from 1 to 100 based on job fit")
    summary: str = Field(description="A 2-3 sentence concise profile summary")
    strengths: List[str] = Field(description="Key qualifications aligning with the job description")
    gaps: List[str] = Field(description="Required job skills or experience missing from the resume")
    interview_questions: List[str] = Field(description="3-4 targeted, role-specific questions to ask this candidate")

# Initialize OpenRouter Client (Expects OPENROUTER_API_KEY environment variable)
client = OpenAI(
    base_url="https://openrouter.ai",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

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

    # 2. Build the AI prompt
    prompt = f"""
    You are an expert HR recruitment assistant. Analyze the candidate's Resume against the provided Job Description.
    
    JOB DESCRIPTION:
    {job_description}
    
    CANDIDATE RESUME:
    {resume_text}
    
    Extract details, compare qualifications, identify strengths/gaps, score the candidate, and generate interview questions.
    """

    try:
        # 3. Call OpenRouter with Structured Outputs configuration
        response = client.beta.chat.completions.parse(
            model='google/gemini-2.5-flash',
            messages=[
                {"role": "user", "content": prompt}
            ],
            response_format=CandidateEvaluation,
            temperature=0.2,
        )
        
        # 4. Return parsed message model object directly to the frontend
        return response.choices[0].message.parsed
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenRouter API Error: {str(e)}")
