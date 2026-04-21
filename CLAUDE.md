# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Quick-Segment is a segmentation API that accepts natural language queries and returns segmentation masks from the most relevant classes across multiple datasets and models.

**Pipeline:** `query → semantic matching (sentence transformers) → class IDs → YOLO inference → masks + metadata`

## Core Architecture

```
api.py           FastAPI endpoints (/segment, /classes, /health)
matcher.py       Semantic class matching via sentence transformers (all-mpnet-base-v2)
inference.py     YOLO segmentation wrapper (yolo26n-seg.pt), lazy model loading
datasets/coco.py COCO 80-class ID map (class name → integer ID)
```

**Key patterns:**
- `Matcher.register(dataset_name, class_list)` → `Matcher.build()` → `Matcher.get_matching_classes(query, threshold)`
- `Inferencer.predict(image, matches, conf)` where `matches` is `list[tuple[class_name, dataset]]`
- Models are loaded lazily per dataset in `Inferencer._models` dict
- Image dimensions are preserved when extracting contours (model outputs 640x640, resized back via `cv2.resize` with `INTER_NEAREST`)

## Commands

```bash
# Run the API (auto-downloads YOLO weights on first use)
uvicorn api:app --reload

# Run all tests
python -m pytest

# Run a single test file
python -m pytest tests/test_matcher.py -v

# Run matcher tests only
python -m pytest tests/test_matcher.py -v
```

## Configuration

| Parameter   | Default | Description                                      |
|-------------|---------|--------------------------------------------------|
| `threshold` | `0.3`   | Cosine similarity threshold for class matching    |
| `conf`      | `0.5`   | YOLO detection confidence threshold              |

## Supported Datasets

| Dataset | Model           | Classes |
|---------|-----------------|---------|
| COCO    | yolo26n-seg.pt  | 80      |

## Data Flow for `/segment`

1. Image uploaded → decoded via PIL as RGB
2. `matcher.get_matching_classes(query, threshold)` returns `list[tuple[class_name, dataset]]`
3. For each unique dataset in matches, group class IDs and run `YOLO.predict(source=image, classes=class_ids, conf=conf)`
4. Extract contours from masks via `cv2.findContours` (largest contour only, resized to original image dimensions)
5. Return combined results with `class_name`, `dataset`, `confidence`, `bbox`, `mask`
