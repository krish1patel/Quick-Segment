from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
# from typing import Any, Optional



class Matcher:
    """A class to match inputted vocabulary to a set of classes in a dataset"""

    def __init__(self) -> None:
        # self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.model = SentenceTransformer("all-mpnet-base-v2")
        self.datasets: dict[str, list[str]] = {}    # key: dataset_name, value: class_name
        self.registry = []
        self.embeddings = None


    def register(self, dataset_name: str, dataset_classes: list[str]) -> None:
        """
        Add a dataset to the matcher

        Args:
            dataset_name (str): name of the dataset being added (e.g. coco)
            dataset_classes (list[str]): a list containing strings of classes in the dataset

        Returns:
            None
        """
        self.datasets[dataset_name] = dataset_classes
        for item in dataset_classes:
            self.registry.append((item, dataset_name))
        
    def build(self) -> None:
        """
        Build embeddings for all uploaded classes

        Args: 
            None

        Returns:
            None
        """
        self.class_names = [class_name for class_name, _ in self.registry]
        self.embeddings = self.model.encode(self.class_names)



    def get_nearest_class(self, query: str) -> tuple[str, str]:
        """
        Get the most similar class from uploaded datasets compared to query.

        Args:
            query (str): input string to find nearest class for

        Returns:
            (str, str): (class_name, dataset).
        """
        if self.embeddings is None:
            raise RuntimeError("Call build() before get_nearest_class()")
    
        query_embedding = self.model.encode([query])
        scores = cosine_similarity(query_embedding, self.embeddings)[0]
        # scores = self.model.similarity(query_embedding, self.embeddings)[0]
        best_idx = np.argmax(scores)  # no need for argsort if you just want the top one
    
        return self.registry[best_idx]

    def get_matching_classes(self, query: str, threshold: float = .3) -> list[tuple[str, str]]:
        """
        Get all matching classes from uploaded datasets compared to query using threshold.

        Args:
            query (str): input string to find nearest class for
            threshold (float): value from 0 to 1 representing the 
                minimum cosine similarity for a class to be returned

        Returns:
            list[tuple[str, str]]: list of tuples containing (class_name, dataset)
        """
        if self.embeddings is None:
            raise RuntimeError("Call build() before get_matching_classes()")
        
        query_embedding = self.model.encode([query])
        scores = cosine_similarity(query_embedding, self.embeddings)[0]

        matches = []

        for i, score in enumerate(scores):
            if score >= threshold:
                matches.append(self.registry[i])

        return matches
