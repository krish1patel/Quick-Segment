import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from matcher import Matcher

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

test_cases = [
    ("find all vehicles", ["car", "truck", "bus", "motorcycle", "bicycle"]),
    ("show me animals", ["cat", "dog", "bird", "horse", "cow", "sheep", "elephant", "bear", "zebra", "giraffe"]),
    ("food items", ["banana", "apple", "sandwich", "pizza", "donut", "cake", "hot dog", "carrot", "broccoli"]),
    ("electronics", ["tv", "laptop", "cell phone", "keyboard", "mouse", "remote"]),
]

matcher = Matcher()
matcher.register("coco", COCO_CLASSES)
matcher.build()

for threshold in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
    print(f"\n--- threshold: {threshold} ---")
    for query, expected in test_cases:
        results = [cls for cls, _ in matcher.get_matching_classes(query, threshold)]
        hits = [r for r in results if r in expected]
        misses = [r for r in results if r not in expected]
        print(f"query: '{query}' | hits: {hits} | false positives: {misses}")
