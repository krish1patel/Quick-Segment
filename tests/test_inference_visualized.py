import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "/Users/krishpatel/Documents/Projects/seg")))
import PIL.Image as Image
import numpy as np


from matcher import Matcher
from inference import Inferencer

from datasets.coco import COCO_CLASS_IDS

COCO_CLASSES = list(COCO_CLASS_IDS.keys())

matcher = Matcher()
matcher.register("coco", COCO_CLASSES)
matcher.build()

inferencer = Inferencer()

test_image = Image.open("/Users/krishpatel/Documents/Projects/seg/tests/media/bus.jpg")

test_query = "person"

matches = matcher.get_matching_classes(test_query)

results = inferencer.predict(test_image, matches)

print(results)



def visualize_results(image: Image.Image, results: list[dict], output_path: str = "vis_output.jpg", save: bool = False):
    import PIL.ImageDraw as ImageDraw

    vis = image.convert("RGBA")
    overlay = Image.new("RGBA", vis.size, (0, 0, 0, 0))

    palette = [
        (255, 50,  50,  120),
        (50,  200, 50,  120),
        (50,  100, 255, 120),
        (255, 200, 0,   120),
        (200, 0,   255, 120),
    ]

    for i, det in enumerate(results):
        color = palette[i % len(palette)]
        solid_color = color[:3]
        contour = det["mask"]

        # ── Mask ──────────────────────────────────────────────────────
        if contour:
            pts = [tuple(pt) for pt in contour]
            mask_draw = ImageDraw.Draw(overlay)
            mask_draw.polygon(pts, fill=color, outline=solid_color + (255,))

        # ── BBox ──────────────────────────────────────────────────────
        x1, y1, x2, y2 = det["bbox"]
        bbox_draw = ImageDraw.Draw(overlay)
        for thickness in range(3):
            bbox_draw.rectangle(
                [x1 - thickness, y1 - thickness, x2 + thickness, y2 + thickness],
                outline=solid_color + (255,)
            )

        # ── Label ─────────────────────────────────────────────────────
        label = f"{det['class_name']} {det['confidence']:.2f}"
        bbox_draw.rectangle([x1, y1 - 18, x1 + len(label) * 7, y1], fill=solid_color + (220,))
        bbox_draw.text((x1 + 2, y1 - 16), label, fill=(255, 255, 255, 255))

    result_img = Image.alpha_composite(vis, overlay).convert("RGB")
    if save:
        result_img.save(output_path)
        print(f"Saved visualization to {output_path}")
    result_img.show()

visualize_results(test_image, results)