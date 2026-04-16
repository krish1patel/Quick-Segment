# SegMatch — Project Plan

## Overview

An API that takes a natural language query, semantically matches it to known segmentation classes across multiple datasets, runs the appropriate YOLO model(s), and returns segmentation mask data. Free and paid tiers enforced via Supabase.

---

## Architecture

```
user query → matcher → class(es) + dataset(s) → YOLO inference → masks + metadata
```

---

## Stack

- **Inference:** YOLO26-seg (Ultralytics) — pretrained COCO weights, auto-downloaded
- **Semantic matching:** `all-mpnet-base-v2` via sentence-transformers + cosine similarity (sklearn)
- **API:** FastAPI
- **Rate limiting / API keys:** Supabase
- **Hosting:** Railway or Fly.io (free tier)
- **Model weights:** Lazy-loaded on first request, cached in memory

---

## Dataset & Model Registry

- Multiple datasets can be registered into the Matcher
- Each dataset maps to a corresponding YOLO model
- Class overlap across datasets resolved by **model hierarchy** (per-class priority, not per-model)
- Example registry:

| Dataset    | Model       | Notes                                                   |
| ---------- | ----------- | ------------------------------------------------------- |
| COCO       | yolo26s-seg | 80 general classes, highest priority for shared classes |
| LVIS       | yolo26-lvis | 1200+ classes, long-tail coverage                       |
| (more TBD) | ...         | Domain-specific (medical, aerial, etc.)                 |

---

## Components

### ✅ 1. `matcher.py` — COMPLETE

Semantic class matching layer.

**Key design decisions:**

- Dataset-agnostic — any number of datasets can be registered
- Uses `all-mpnet-base-v2` sentence transformer for embeddings
- Embeddings built once via `build()` after all datasets registered
- Cosine similarity against flat registry of `(class_name, dataset)` tuples
- Returns all matches above a threshold (default: `0.3`)
- `np.argmax` / score filtering — no loops, fully vectorized

**Interface:**

```python
matcher = Matcher()
matcher.register("coco", COCO_CLASSES)
matcher.build()

matches = matcher.get_matching_classes("find all vehicles")
# [("car", "coco"), ("truck", "coco"), ("bus", "coco")]
```

---

### 🔲 2. `inference.py` — NEXT

YOLO inference wrapper.

**Responsibilities:**

- Load and cache YOLO models per dataset (lazy load on first use)
- Accept an image + list of `(class_name, dataset)` tuples from matcher
- Filter YOLO results to only the matched classes
- Return mask data, bounding boxes, confidence scores per detection

**Interface (planned):**

```python
results = run_inference(image, matches)
# returns list of detections with masks, boxes, scores, class names
```

---

### 🔲 3. `api.py` — TODO

FastAPI wrapper.

**Endpoints (planned):**

- `POST /segment` — main endpoint, accepts image + query string
- `GET /classes` — list all available classes across registered datasets
- `GET /health` — health check

**Free vs paid tier:**

- Free: rate limited, smaller model variant (yolo26n-seg)
- Paid: higher rate limit, larger model (yolo26s-seg or above)
- API key tracking via Supabase

---

## Cost Target

- **Inference:** CPU-only for free tier (acceptable ~500ms latency), GPU via Modal for paid
- **Embedding:** one-time encode at startup, zero per-request cost
- **Hosting:** Railway/Fly.io free tier
- **Target monthly cost at low traffic: ~$0–5**

---

## Threshold Calibration

Swept thresholds 0.2–0.5 against COCO classes. Results:

| Threshold | Hits             | False Positives      |
| --------- | ---------------- | -------------------- |
| 0.2       | High             | Very high            |
| 0.25      | High             | High                 |
| **0.3**   | **Good**         | **Low — sweet spot** |
| 0.35      | Drops valid hits | Very low             |
| 0.4+      | Too restrictive  | Near zero            |

**Default threshold: 0.3** (tunable per request)

---

## Directory Structure

```
segmatch/
  matcher.py        ✅ done
  inference.py      🔲 next
  api.py            🔲 todo
  datasets/
    coco.py         🔲 class lists
  tests/
    test_matcher.py ✅ done
    test_inference.py 🔲 todo
  PLAN.md
```
