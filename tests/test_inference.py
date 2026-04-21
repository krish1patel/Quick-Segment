import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "/Users/krishpatel/Documents/Projects/seg")))
import PIL.Image as Image

from matcher import Matcher
from inference import Inferencer

from datasets.coco import COCO_CLASS_IDS

COCO_CLASSES = list(COCO_CLASS_IDS.keys())

matcher = Matcher()
matcher.register("coco", COCO_CLASSES)
matcher.build()

inferencer = Inferencer()

test_image = Image.open("/Users/krishpatel/Documents/Projects/seg/tests/media/bus.jpg")

test_query = "vehicles"

matches = matcher.get_matching_classes(test_query)

results = inferencer.predict(test_image, matches)

print(results)