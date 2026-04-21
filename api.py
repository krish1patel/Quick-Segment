from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
import PIL.Image as Image
import io

from matcher import Matcher
from inference import Inferencer
from datasets.coco import COCO_CLASS_IDS

# ── Startup ───────────────────────────────────────────────────────────────────

app = FastAPI(title="QuickSegment", version="0.1.0")

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


@app.post("/segment")
async def segment(
    file: UploadFile = File(...),
    query: str = Query(..., description="Natural language query, e.g. 'vehicles'"),
    conf: float = Query(0.5, ge=0.0, le=1.0),
    threshold: float = Query(0.3, ge=0.0, le=1.0),
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