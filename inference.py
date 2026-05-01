from ultralytics import YOLO, YOLOE
from datasets.coco import COCO_CLASS_IDS

# import cv2
# import numpy as np

from datasets.yoloe import YOLOE_CLASS_IDS

MODEL_MAP = { 
    "coco": "yolo26n-seg.pt",
    "yoloe": "yoloe-26s-seg-pf.pt"
}

DATASET_ID_MAP = {
    "coco": COCO_CLASS_IDS,
    "yoloe": YOLOE_CLASS_IDS
}

class Inferencer:
    """
    Run yolo segmentation inference across several datasets
    """

    def __init__(self) -> None:
        self._models: dict[str, YOLO] = {"coco": YOLO("yolo26n-seg.pt"), "yoloe": YOLOE("yoloe-26s-seg-pf.pt")}

    # @staticmethod
    # def _extract_contours(mask: np.ndarray, orig_w: int, orig_h: int) -> list[list[int]]:
    #     # Resize from model space (640x640) back to original image dimensions
    #     mask_resized = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    #     binary = (mask_resized > 0.5).astype(np.uint8)
    #     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     if not contours:
    #         return []
    #     # Take the largest contour (handles noise/fragments)
    #     largest = max(contours, key=cv2.contourArea)
    #     # Flatten [[x, y], [x, y], ...] → [x, y, x, y, ...]
    #     return largest.reshape(-1, 2).tolist()


    def predict(self, image, matches: list[tuple[str, str]], conf: float = 0.5) -> list[dict]:
        """
        Returns segmentation results on inputted image and matches
        Args:
            image: input image
            matches (list[tuple[str, str]]): list of tuples containing matched (class_name, dataset)
            conf (float): confidence threshold
        Returns:
            list[dict]: combined segmentation results from all models
        """
        datasets: dict[str, list[int]] = {}
        for class_name, dataset in matches:
            if dataset not in MODEL_MAP:
                raise ValueError(f'Dataset {dataset} not currently supported')
            if dataset not in self._models:
                self._models[dataset] = YOLO(MODEL_MAP[dataset])
            
            class_id = DATASET_ID_MAP[dataset][class_name]
            if dataset not in datasets:
                datasets[dataset] = [class_id]
            else:
                datasets[dataset].append(class_id)

        if hasattr(image, "width"):
            orig_w, orig_h = image.width, image.height  # PIL
        else:
            orig_h, orig_w = image.shape[:2]  # numpy/cv2

        output = []
        for dataset, class_ids in datasets.items():
            results = self._models[dataset].predict(source=image, classes=class_ids, conf=conf)
            for result in results:
                if result.masks is None:
                    continue
                masks_xy = result.masks.xy  # list of np.ndarray, each (N, 2), in original image coords
                for i, polygon in enumerate(masks_xy):
                    box = result.boxes[i]
                    class_id = int(box.cls.item())
                    output.append({
                        "class_name": result.names[class_id],
                        "dataset": dataset,
                        "confidence": round(float(box.conf.item()), 3),
                        "bbox": [round(float(x), 2) for x in box.xyxy[0].tolist()],
                        "mask": polygon.round().astype(int).tolist()
                    })

        return output
