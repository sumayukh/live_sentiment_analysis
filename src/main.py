import cv2
# Import custom classes
from facetracker import TrackFace
from sentiment_analysis import AnalyzeSentiment


def app():
    # Instantiate track and analyze objects
    track = TrackFace()
    analyze = AnalyzeSentiment()

    # Initialize video capture from default camera (0)
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, value=1080)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, value=1920)
    if not cam.isOpened():
        print('CAMERA IS NOT ACCESSIBLE')
        return

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                break
            print("Frame captured")
            # Detect faces and get both ROI with bounding boxes
            detected_faces = track.extract_faces(frame)

            for face, (x1, y1, x2, y2) in detected_faces:
                # Analyze each face ROI
                face_label = analyze.extract_data(face)
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # Overlay label above the box
                cv2.putText(
                    frame,
                    f"{face_label}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            cv2.imshow("Live Sentiment Analysis", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cam.release()
        cv2.destroyAllWindows()

# main
if __name__ == "__main__":
    app()