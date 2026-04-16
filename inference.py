from ultralytics import YOLO
from datasets.coco import COCO_CLASS_IDS

MODEL_MAP = { 
    "coco": "yolo26n-seg.pt" #,
    # "lvis": "yolo26n-lvis.pt"
}

DATASET_ID_MAP = {
    "coco": COCO_CLASS_IDS
}

class Inferencer:
    """
    Run yolo segmentation inference across several datasets
    """

    def __init__(self) -> None:
        self._models: dict[str, YOLO] = {"coco": YOLO("yolo26n-seg.pt")}


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

        output = []
        for dataset, class_ids in datasets.items():
            results = self._models[dataset].predict(source=image, classes=class_ids, conf=conf)
            for result in results:
                if result.masks is None:
                    continue
                for i, mask in enumerate(result.masks.data):
                    box = result.boxes[i]
                    class_id = int(box.cls.item())
                    output.append({
                        "class_name": result.names[class_id],
                        "dataset": dataset,
                        "confidence": round(float(box.conf.item()), 3),
                        "bbox": [round(float(x), 2) for x in box.xyxy[0].tolist()],
                        "mask": mask.cpu().numpy().tolist()
                    })

        return output
