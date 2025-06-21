import cv2
# Import custom classes
from facetracker import TrackFace
from sentiment_analysis import AnalyzeSentiment


def app():
    metrics = ['emotion']
    # Instantiate track and analyze objects
    track = TrackFace()
    analyze = AnalyzeSentiment(actions=metrics)

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
                analysis = analyze.extract_data(face)
                # Draw bounding circle
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                width = abs(x2 - x1)
                height = abs(y2 - y1)
                radius = int(max(width, height)) + 10
                cv2.circle(frame, center, radius, (255, 255, 255), 2)
                # Overlay label beside the circle

                text_y = center[1] - radius - 20
                for key, val in analysis.items():
                    formatted_key = key.replace('_', ' ').split(' ')[1].capitalize()
                    formatted_val = val.capitalize()
                    text = f"{formatted_key}: {formatted_val}"
                    # Get text size
                    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 0.9, 2)
                    text_width = text_size[0]

                    # Calculate position: horizontally centered, vertically just below the circle
                    text_x = center[0] - text_width // 2

                    cv2.putText(
                        frame,
                        text,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.9,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
            cv2.imshow("Live Sentiment Analysis", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cam.release()
        cv2.destroyAllWindows()

# main
if __name__ == "__main__":
    app()