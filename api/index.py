import os
import io
import uuid
from typing import List, Dict, Optional, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Document Parser Libraries
from pypdf import PdfReader
from docx import Document
from PIL import Image

# Modern Gemini API Wrapper SDK
from google import genai
from google.genai import types

app = FastAPI(title="HireAI Recruiting Dashboard Backend API")

# Configure CORS to communicate seamlessly with your custom HTML dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the GenAI Client (Make sure GEMINI_API_KEY environment variable is set)
client = genai.Client()

# --- Volatile In-Memory Dashboard Mock Databases ---
# Holds current application state metrics during active runtime runtime loops
JOB_ROLES_DB: Dict[str, Dict[str, Any]] = {}
CANDIDATES_DB: Dict[str, Dict[str, Any]] = {}


# --- Pydantic Data Structures ---

class JobRoleCreate(BaseModel):
    title: str
    department: str
    requirements: str

class CandidateProfile(BaseModel):
    name: str = Field(..., description="Extract candidate full name. Default to 'Unknown Applicant' if missing.")
    skills: List[str] = Field(default_factory=list, description="Array of technical or soft skills discovered.")
    experience_years: float = Field(0.0, description="Total summary of relevant experience in fractional or whole numbers.")
    education: List[str] = Field(default_factory=list, description="Key degrees, fields of study, or universities.")
    certifications: List[str] = Field(default_factory=list, description="Professional licenses or training credentials.")

class CandidateEvaluationResult(BaseModel):
    candidate_profile: CandidateProfile
    match_score: int = Field(..., description="An objective scoring parameter grading alignment against rules between 0 and 100.")
    summary: str = Field(..., description="A punchy overview summarizing core competencies.")
    strengths: List[str] = Field(default_factory=list, description="Bullet metrics highlighting candidate advantages.")
    gaps: List[str] = Field(default_factory=list, description="Qualifications or technical items noted as missing.")
    interview_questions: List[str] = Field(default_factory=list, description="3-5 customized structured behavioral or role interview questions.")


# --- Document & Image Reading Parsers ---

def extract_text_from_bytes(file_bytes: bytes, filename: str) -> str:
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
            ocr_prompt = "Transcribe all printed text and tabular experience formatting from this resume file exactly."
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[image, ocr_prompt]
            )
            return response.text or ""
        else:
            raise HTTPException(status_code=400, detail="Supported formats strictly include PDF, DOCX, PNG, or JPG.")
    except Exception as e:
        if isinstance(e, HTTPException): raise e
        raise HTTPException(status_code=422, detail=f"Failed to cleanly extract text content from file asset: {str(e)}")


# --- Primary Dashboard Routes ---

@app.get("/api/dashboard/metrics")
async def get_dashboard_metrics():
    """
    Returns summary counters mapping directly to top metric rows in the UI
    """
    total_candidates = len(CANDIDATES_DB)
    active_roles = len(JOB_ROLES_DB)
    
    # Calculate ongoing total evaluations completed
    evaluations_count = sum(1 for c in CANDIDATES_DB.values() if c.get("evaluated"))
    
    # Calculate accurate moving match average parameters
    valid_scores = [c["match_score"] for c in CANDIDATES_DB.values() if c.get("evaluated") and c.get("match_score") is not None]
    avg_score = f"{round(sum(valid_scores) / len(valid_scores))}%" if valid_scores else "—"

    # Compile array list containing top matches sorted by descending order parameters
    ranked_candidates = [
        {
            "id": cid,
            "name": c["name"],
            "score": c["match_score"],
            "experience": c["experience_years"],
            "role_title": c["role_title"]
        }
        for cid, c in CANDIDATES_DB.items() if c.get("evaluated")
    ]
    ranked_candidates.sort(key=lambda x: x["score"], reverse=True)

    return {
        "counters": {
            "total_candidates": total_candidates,
            "active_roles": active_roles,
            "evaluations": evaluations_count,
            "avg_match_score": avg_score
        },
        "top_ranked": ranked_candidates[:5] # Return top 5 profiles
    }


@app.post("/api/roles")
async def create_job_role(role: JobRoleCreate):
    """
    Registers a new target vacancy standard layout
    """
    role_id = str(uuid.uuid4())[:8]
    JOB_ROLES_DB[role_id] = {
        "id": role_id,
        "title": role.title,
        "department": role.department,
        "requirements": role.requirements
    }
    return {"message": "Job role established safely.", "role_id": role_id}


@app.get("/api/roles")
async def list_job_roles():
    return list(JOB_ROLES_DB.values())


@app.post("/api/candidates/upload")
async def upload_candidate_resume(
    name: str = Form(...),
    role_id: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Receives raw resume file attachments, extracts plaintext, and generates background profiles
    """
    if role_id not in JOB_ROLES_DB:
        raise HTTPException(status_code=404, detail="Target job role tracking reference not found.")
        
    file_bytes = await file.read()
    extracted_text = extract_text_from_bytes(file_bytes, file.filename or "resume.pdf")
    
    candidate_id = str(uuid.uuid4())[:8]
    CANDIDATES_DB[candidate_id] = {
        "id": candidate_id,
        "name": name,
        "role_id": role_id,
        "role_title": JOB_ROLES_DB[role_id]["title"],
        "resume_text": extracted_text,
        "evaluated": False,
        "match_score": None
    }
    
    return {"message": "Candidate resume file parsed successfully.", "candidate_id": candidate_id}


@app.post("/api/candidates/{candidate_id}/evaluate", response_model=CandidateEvaluationResult)
async def evaluate_candidate(candidate_id: str):
    """
    Executes advanced structured evaluation loops using the Gemini analytical context engine
    """
    candidate = CANDIDATES_DB.get(candidate_id)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate record not found.")
        
    role = JOB_ROLES_DB.get(candidate["role_id"])
    if not role:
        raise HTTPException(status_code=404, detail="Linked job position profile missing.")

    analysis_prompt = f"""
    You are HireAI's core executive recruitment processing assistant engine. 
    Analyze the provided candidate resume content and compare alignment against target role specifications.
    
    [TARGET ROLE SPECIFICATION]
    Title: {role['title']}
    Requirements: {role['requirements']}

    [APPLICANT RESUME CONTENT]
    {candidate['resume_text']}

    Complete parsing accuracy parameters and structure values exactly within the given response schema format.
    """

    try:
        # Utilize deep structural processing parameters via strict type tracking schemas
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=analysis_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=CandidateEvaluationResult,
                temperature=0.15, # Tight tracking for precise profile mapping loops
            )
        )
        
        # Hydrate JSON result maps straight through schema validations
        evaluation = CandidateEvaluationResult.model_validate_json(response.text)
        
        # Commit updated tracking parameters into working runtime memory maps
        CANDIDATES_DB[candidate_id].update({
            "evaluated": True,
            "name": evaluation.candidate_profile.name,
            "match_score": evaluation.match_score,
            "experience_years": evaluation.candidate_profile.experience_years,
            "details": evaluation.model_dump()
        })
        
        return evaluation

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI context grading evaluation process faulted: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    # Execute backend loops locally matching frontend expectations
    uvicorn.run(app, host="0.0.0.0", port=8000)
