from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from extract_events import extract_events


app = FastAPI(
    title="Health Assistant API",
    version="1.0.0",
)


class PatientTextRequest(BaseModel):
    text: str


@app.get("/")
def root():
    return {"status": "API is running"}


@app.post("/extract-events")
def extract_patient_events(request: PatientTextRequest):
    try:
        return extract_events(request.text)

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Failed to extract patient events.",
        ) from error
