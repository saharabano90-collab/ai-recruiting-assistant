import os
import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from openai import OpenAI
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
    base_url="https://openrouter.ai/api/v1",
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

    # 2. Build the AI prompt embedding the required Pydantic schema structure
    # Pydantic's model_json_schema() exports a clean JSON schema structure OpenRouter models can understand
    json_schema_str = json.dumps(CandidateEvaluation.model_json_schema(), indent=2)
    
    prompt = f"""
    You are an expert HR recruitment assistant. Analyze the candidate's Resume against the provided Job Description.
    
    JOB DESCRIPTION:
    {job_description}
    
    CANDIDATE RESUME:
    {resume_text}
    
    Task: Extract details, compare qualifications, identify strengths/gaps, score the candidate, and generate interview questions.
    You MUST output valid JSON conforming exactly to this schema:
    {json_schema_str}
    """

    try:
        # 3. Request completion using OpenRouter's baseline JSON object validation
        response = client.chat.completions.create(
            model='google/gemini-2.5-flash',
            messages=[
                {
                    "role": "system", 
                    "content": "You are a precise data extraction system. You must return RAW JSON matching the requested schema. No conversational filler, no markdown code blocks."
                },
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}, 
            temperature=0.1, # Reduced temperature for stricter formatting adherence
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # 4. Safe Parsing Cleanup
        # If the model ignored the system prompt and wrapped the response in a markdown block, strip it.
        if response_text.startswith("```"):
            response_text = response_text.strip("```").strip("json").strip()
            
        parsed_json = json.loads(response_text)
        
        # Validate the keys strictly using Pydantic before sending it to the frontend
        return CandidateEvaluation(**parsed_json)
        
    except json.JSONDecodeError as je:
        raise HTTPException(
            status_code=502, 
            detail=f"OpenRouter returned unparseable text: {response_text}. Error: {str(je)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"OpenRouter runtime error: {str(e)}"
        )
