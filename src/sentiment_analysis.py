import numpy as np
from deepface import DeepFace


# Sentiment Analysis class
class AnalyzeSentiment:
    # Analyze facial expressions using DeepFace
    def __init__(self, actions: list[str] = ["emotion"]):
        self.actions = actions
        self.target_size = (224, 224)

    def extract_data(self, face: np.ndarray) -> dict:
        # Detect emotion label for a given face
        if face is None or face.size == 0:
            print("Invalid face region.")
            return {}
        try:
            output = DeepFace.analyze(
                img_path=face,
                actions=self.actions,
                enforce_detection=False,
                detector_backend='skip',
                align=False,
            )[0]
            print('OUTPUT ===>', output)
            output_metrics = [
                f'dominant_{key}' if key in {"emotion", 'race'} else key for key in self.actions
            ]
            analysis = {key: output.get(key, 'unknown') for key in output_metrics}
            print('OUTPUT_METRICS ===>', output_metrics)
            print('ANALYSIS ===>', analysis)
            return analysis
        except Exception as e:
            print(f"DeepFace analysis error: {e}")
            return {}