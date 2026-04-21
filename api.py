from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import PIL.Image as Image
import io
import os
from fastapi.responses import JSONResponse

from matcher import Matcher
from inference import Inferencer
from datasets.coco import COCO_CLASS_IDS

# ── Startup ───────────────────────────────────────────────────────────────────

app = FastAPI(title="QuickSegment", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/tests", StaticFiles(directory="tests"), name="tests")
app.mount("/media", StaticFiles(directory="./tests/media"), name="media")

COCO_CLASSES = list(COCO_CLASS_IDS.keys())

matcher = Matcher()
matcher.register("coco", COCO_CLASSES)
matcher.build()

inferencer = Inferencer()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/classes")
def classes():
    return matcher.datasets

@app.get("/media-list")
def list_media():
    media_dir = "./tests/media"
    files = [
        f for f in os.listdir(media_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    return JSONResponse(files)


@app.post("/segment")
async def segment(
    file: UploadFile = File(...),
    query: str = Query(..., description="Natural language query, e.g. 'vehicles'"),
    conf: float = Query(0.4, ge=0.0, le=1.0),
    threshold: float = Query(0.5, ge=0.0, le=1.0),
):
    # Validate image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image")

    # Match classes
    matches = matcher.get_matching_classes(query, threshold=threshold)
    if not matches:
        return JSONResponse(content={"query": query, "matches": [], "detections": []})

    # Run inference
    try:
        results = inferencer.predict(image, matches, conf=conf)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return JSONResponse(content={
        "query": query,
        "matches": [{"class_name": c, "dataset": d} for c, d in matches],
        "detections": results,
    })