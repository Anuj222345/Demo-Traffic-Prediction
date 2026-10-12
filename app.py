from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import os

app = Flask(__name__)

# Load trained YOLOv8 model for 15 classes (Self-Driving Cars v6)
# Use your custom trained weights if you have them, otherwise it falls back to yolov8n.pt
MODEL_PATH = 'best.pt' if os.path.exists('best.pt') else 'yolov8n.pt'
model = YOLO(MODEL_PATH)

CLASSES = [
    'Green Light', 'Red Light', 'Speed Limit 10', 'Speed Limit 100',
    'Speed Limit 110', 'Speed Limit 120', 'Speed Limit 20', 'Speed Limit 30',
    'Speed Limit 40', 'Speed Limit 50', 'Speed Limit 60', 'Speed Limit 70',
    'Speed Limit 80', 'Speed Limit 90', 'Stop'
]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({'error': 'Invalid image file'}), 400

    # Run YOLO inference
    results = model(img, conf=0.45)
    detections = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            detections.append({
                'class_id': cls_id,
                'class_name': CLASSES[cls_id] if cls_id < len(CLASSES) else 'Unknown',
                'confidence': round(conf * 100, 2),
                'bbox': [round(c, 2) for c in xyxy]
            })

    return jsonify({'success': True, 'detections': detections})


if __name__ == '__main__':
    app.run(debug=True)

