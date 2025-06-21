import cv2
import mediapipe as mp
import numpy as np


# Face Tracker class

class TrackFace:
    # Detects faces using MediaPipe Face Mesh and returns cropped face ROIs.

    def __init__(self, min_detection_confidence: int = 50, min_tracking_confidence: int = 50):
        self.mp_face = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_mesh = self.mp_face.FaceMesh(
            min_detection_confidence, min_tracking_confidence, refine_landmarks=True
        )

    # Processes the input frame, draws face mesh overlay (optional), and returns a list of cropped face images.

    def extract_faces(self, frame: np.ndarray) -> list[np.ndarray]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = self.face_mesh.process(rgb_frame)
        faces = []

        if output.multi_face_landmarks:
            h, w = frame.shape[:2]

            pad = 10
            for face in output.multi_face_landmarks:
                # Compute bounding box of landmarks

                xlist = [ele.x for ele in face.landmark]
                ylist = [ele.y for ele in face.landmark]
                x1, x2 = int(min(xlist) * w), int(max(xlist) * w)
                y1, y2 = int(min(ylist) * h), int(max(ylist) * h)

                # Custom padding

                x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                x2, y2 = min(w - 1, x2 + pad), min(h - 1, y2 + pad)

                # Make square
                box_w, box_h = x2 - x1, y2 - y1
                diff = abs(box_w - box_h)
                if box_w > box_h:
                    y2 = min(h - 1, y2 + diff)
                else:
                    x2 = min(w - 1, x2 + diff)

                # Region of interest

                roi = frame[y1:y2, x1:x2]
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                if roi.size > 0:
                    faces.append((roi, (x1, y1, x2, y2)))

        return faces