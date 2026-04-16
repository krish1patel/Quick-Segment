# Quick-Segment

A segmentation API that accepts natural language queries and returns segmentation masks from the most relevant classes across multiple datasets and models.

## How It Works

```
query → semantic matching → class IDs → YOLO inference → masks + metadata
```

1. User submits an image and a natural language query (e.g. "find all the vehicles")
2. The query is embedded and compared against all known classes using cosine similarity
3. Matching classes are grouped by dataset and routed to the appropriate YOLO model
4. Results are combined and returned as a list of detections with masks, bounding boxes, and confidence scores

## Project Structure

```
segmatch/
  matcher.py          # semantic class matching via sentence transformers
  inference.py        # YOLO inference wrapper, lazy model loading
  api.py              # FastAPI endpoints
  datasets/
    __init__.py
    coco.py           # COCO class list and class→ID map
  tests/
    test_matcher.py
    test_inference.py
```

## Supported Datasets

| Dataset | Model       | Classes            |
| ------- | ----------- | ------------------ |
| COCO    | yolo26n-seg | 80 general classes |

More datasets coming soon.

## API

### `POST /segment`

Submit an image and query, returns segmentation results.

**Request**

```json
{
  "query": "find all the vehicles",
  "image": "<base64 encoded image>"
}
```

**Response**

```json
[
  {
    "class_name": "car",
    "dataset": "coco",
    "confidence": 0.91,
    "bbox": [x1, y1, x2, y2],
    "mask": [[...], ...]
  }
]
```

### `GET /classes`

Returns all available classes across registered datasets.

### `GET /health`

Health check.

## Setup

**Requirements:** Python 3.11+, pip

```bash
# clone and install
git clone https://github.com/yourname/quick-segment
cd quick-segment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Run the API**

```bash
uvicorn api:app --reload
```

YOLO model weights are downloaded automatically on first use.

## Configuration

| Parameter   | Default | Description                                    |
| ----------- | ------- | ---------------------------------------------- |
| `threshold` | `0.3`   | Cosine similarity threshold for class matching |
| `conf`      | `0.5`   | YOLO detection confidence threshold            |

## Tiers

|          | Free         | Paid         |
| -------- | ------------ | ------------ |
| Requests | Rate limited | Higher limit |
| Model    | yolo26n-seg  | yolo26s-seg  |

## Dependencies

- [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLO26 inference
- [sentence-transformers](https://www.sbert.net/) — semantic class matching
- [FastAPI](https://fastapi.tiangolo.com/) — API framework
- [scikit-learn](https://scikit-learn.org/) — cosine similarity
