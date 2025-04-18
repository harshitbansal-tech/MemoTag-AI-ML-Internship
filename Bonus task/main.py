from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import os
import shutil
import numpy as np
from voice_utils import (
    init_db, save_embedding_to_db, load_embedding_from_db,
    record_audio, load_audio_from_npy, denoise_audio,
    extract_features, compare_embeddings, is_live, log_history,
    get_user_history
)

app = FastAPI()
init_db()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
THRESHOLD = 0.75


@app.post("/register")
async def register(username: str = Form(...), file: UploadFile = File(...)):
    if not file.filename.endswith(".npy"):
        raise HTTPException(status_code=400, detail="Only .npy files are supported.")

    file_path = os.path.join(UPLOAD_DIR, f"{username}_register.npy")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    audio = load_audio_from_npy(file_path.replace(".npy", ""))
    if not is_live(audio):
        raise HTTPException(status_code=400, detail="Liveness check failed.")

    audio = denoise_audio(audio)
    embedding = extract_features(audio)
    save_embedding_to_db(username, embedding)
    return {"status": "success", "message": f"Voice registered for user '{username}'."}


@app.post("/authenticate")
async def authenticate(username: str = Form(...), file: UploadFile = File(...)):
    if not file.filename.endswith(".npy"):
        raise HTTPException(status_code=400, detail="Only .npy files are supported.")

    file_path = os.path.join(UPLOAD_DIR, f"{username}_auth.npy")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    ref_embed = load_embedding_from_db(username)
    if ref_embed is None:
        raise HTTPException(status_code=404, detail="User not found.")

    audio = load_audio_from_npy(file_path.replace(".npy", ""))
    if not is_live(audio):
        log_history(username, 0.0, False, "Liveness failed")
        return JSONResponse(content={"authenticated": False, "reason": "Liveness check failed"}, status_code=403)

    audio = denoise_audio(audio)
    test_embed = extract_features(audio)
    score = compare_embeddings(ref_embed, test_embed)
    success = score >= THRESHOLD

    log_history(username, float(score), success, "Authenticated" if success else "Failed")

    return {
        "username": username,
        "similarity_score": round(score, 4),
        "authenticated": success
    }


@app.get("/history/{username}")
def get_history(username: str):
    history = get_user_history(username)
    if not history:
        raise HTTPException(status_code=404, detail="No history found for user.")
    return {"user": username, "history": history}
