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

matcher = Matcher()
matcher.register("coco", COCO_CLASSES)
matcher.build()

print('\n\n')
print('Single Class Matching:')
print("find all the vehicles")
print(matcher.get_nearest_class("find all the vehicles"))
print('\n\n')
print('I want to see animals')
print(matcher.get_nearest_class("I want to see animals"))
print('\n\n')
print('show me food items')
print(matcher.get_nearest_class("show me food items"))


print('\n\n')
print('Threshold Based Class Matching:')
print("find all the vehicles")
print(matcher.get_matching_classes("find all the vehicles", threshold=.35))
print('\n\n')
print('I want to see animals')
print(matcher.get_matching_classes("I want to see animals", threshold=.35))
print('\n\n')
print('show me food items')
print(matcher.get_matching_classes("show me food items", threshold=.35))
print('\n\n')
print('edible items')
print(matcher.get_matching_classes("edible items", threshold=.35))
