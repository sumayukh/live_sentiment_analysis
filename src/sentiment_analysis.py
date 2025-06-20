import numpy as np
from deepface import DeepFace


# Sentiment Analysis class
class AnalyzeSentiment:
    # Analyze facial expressions using DeepFace
    def __init__(self, actions: list[str] = ["emotion"]):
        self.actions = actions

    def extract_data(self, face: np.ndarray) -> str:
        # Detect emotion label for a given face
        output = DeepFace.analyze(img_path=face, actions=self.actions, enforce_detection=False)[0]
        return output.get('dominant_emotion', 'unknown')