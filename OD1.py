from flask import Flask, Response, render_template
import cv2
import supervision as sv
from ultralytics import YOLO

# Initialize Flask app
app = Flask(__name__)

# Load the YOLO model
model = YOLO("yolo11n.pt")

# Create annotators
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

def generate_frames():
    cap = cv2.VideoCapture(0)  # Open default webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set height
    cap.set(cv2.CAP_PROP_FPS, 30)  # Set FPS

    if not cap.isOpened():
        raise RuntimeError("Error: Could not access the webcam.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection
        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)

        # Annotate the frame
        annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

        # Encode the frame as JPEG
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()

        # Yield the frame as part of a multipart HTTP response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Start the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
