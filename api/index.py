import io
import os
from typing import List
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pypdf import PdfReader
from docx import Document
from PIL import Image
from google import genai
from google.genai import types

app = FastAPI(title="HireAI API", version="2.0.0")

# Same-origin Vercel deployment normally does not require permissive CORS,
# but this allows direct testing from another frontend during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Gemini client
# -----------------------------
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    # Do not crash the deployment at import time.
    # The API endpoint will return a useful configuration error.
    client = None
else:
    client = genai.Client(api_key=api_key)

# -----------------------------
# Response schemas
# -----------------------------
class CandidateProfile(BaseModel):
    name: str = Field(default="Unknown")
    skills: List[str] = Field(default_factory=list)
    experience_years: float = Field(default=0.0)
    education: List[str] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)

class CandidateEvaluationResult(BaseModel):
    candidate_profile: CandidateProfile
    match_score: int = Field(ge=0, le=100)
    summary: str
    strengths: List[str] = Field(default_factory=list)
    gaps: List[str] = Field(default_factory=list)
    interview_questions: List[str] = Field(default_factory=list)

# -----------------------------
# Document extraction
# -----------------------------
def read_text_from_file_bytes(file_bytes: bytes, filename: str) -> str:
    filename = filename or "resume.pdf"
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    try:
        if ext == "pdf":
            reader = PdfReader(io.BytesIO(file_bytes))
            pages = []
            for page in reader.pages:
                text = page.extract_text() or ""
                if text.strip():
                    pages.append(text)

            return "\n".join(pages).strip()

        if ext == "docx":
            doc = Document(io.BytesIO(file_bytes))
            return "\n".join(
                p.text for p in doc.paragraphs if p.text.strip()
            ).strip()

        if ext in {"png", "jpg", "jpeg", "webp"}:
            if client is None:
                raise HTTPException(
                    status_code=500,
                    detail="GEMINI_API_KEY is not configured in Vercel."
                )
            image = Image.open(io.BytesIO(file_bytes))
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    image,
                    "Extract all readable text from this resume. Return only the text."
                ],
            )
            return (response.text or "").strip()

        raise HTTPException(
            status_code=400,
            detail="Unsupported format. Upload PDF, DOCX, PNG, JPG, or JPEG."
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Could not read resume: {exc}"
        )

# -----------------------------
# Health check
# -----------------------------
@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "gemini_configured": bool(api_key),
    }

# -----------------------------
# Main evaluation endpoint
# -----------------------------
@app.post("/api/evaluate-direct-file")
async def evaluate_direct_file(
    job_requirements: str = Form(...),
    candidate_name: str = Form(""),
    file: UploadFile = File(...),
):
    if client is None:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is missing. Add it in Vercel Project Settings > Environment Variables and redeploy."
        )
    if not job_requirements.strip():
        raise HTTPException(
            status_code=400,
            detail="Job description / requirements are required."
        )

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Resume file is required."
        )
    allowed = {".pdf", ".docx", ".png", ".jpg", ".jpeg", ".webp"}
    extension = (
        "." + file.filename.lower().rsplit(".", 1)[-1]
        if "." in file.filename
        else ""
    )
    if extension not in allowed:
        raise HTTPException(
            status_code=400,
            detail="Unsupported resume format."
        )
    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded resume is empty."
        )
    resume_text = read_text_from_file_bytes(file_bytes, file.filename)

    if not resume_text.strip():
        raise HTTPException(
            status_code=422,
            detail=(
                "No readable text was extracted from this PDF. "
                "If the resume is a scanned/image-only PDF, upload a text-based PDF "
                "or DOCX, or add OCR processing."
            )
        )
    ai_prompt = f"""
You are HireAI, a recruiting evaluation assistant.

Evaluate the candidate ONLY against job-relevant information.

Do not use or infer protected characteristics such as age, gender,
religion, race/ethnicity, marital status, disability, photograph,
nationality, or other non-job-related personal characteristics.

JOB REQUIREMENTS:
{job_requirements}

CANDIDATE NAME PROVIDED BY RECRUITER:
{candidate_name}

RESUME TEXT:
{resume_text}

Return a structured evaluation containing:
- candidate_profile
- match_score from 0 to 100
- concise summary
- strengths
- gaps
- 3 to 5 role-specific interview questions

The score must be based on evidence in the resume and the supplied job
requirements. Do not invent qualifications that are not supported by the resume.
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=ai_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=CandidateEvaluationResult,
                temperature=0.15,
            ),
        )

        if not response.text:
            raise HTTPException(
                status_code=500,
                detail="Gemini returned an empty evaluation."
            )

        result = CandidateEvaluationResult.model_validate_json(response.text)

        # Prefer the recruiter-provided name when available.
        if candidate_name.strip():
            result.candidate_profile.name = candidate_name.strip()

        return result

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"AI evaluation failed: {exc}"
        )
