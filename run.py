import os
import time
import shutil
import warnings

import cv2
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
from PIL import Image

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated."
)

# configs
WEBCAM_SOURCE = 0

# Folder for saving cropped detections
SAVE_DIR = "frame_images"

# Model paths (relative to project root)
YOLO_MODEL_PATH = "results/rotten-fruit-detection.pt"
RESNET_MODEL_PATH = "results/resnet_best_grid.pt"

# YOLO config
CONFIDENCE_THRESHOLD = 0.75
DEVICE_YOLO = "0"  # '0' for first GPU, 'cpu' for CPU
IMAGE_SIZE = 640

# ResNet / classifier config
DEVICE_CLS = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 2  # fresh / rotten
# Adjust if your label ordering is reversed
CLS_IDX_TO_NAME = {0: "fresh", 1: "rotten"}

# save dir
if os.path.exists(SAVE_DIR):
    shutil.rmtree(SAVE_DIR)
os.makedirs(SAVE_DIR, exist_ok=True)

# load resnet
cls_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

resnet_model = models.resnet18(weights=None)
in_features = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(in_features, NUM_CLASSES)
print(f"Loading ResNet classifier from {RESNET_MODEL_PATH} ...")
state = torch.load(RESNET_MODEL_PATH, map_location=DEVICE_CLS)

if isinstance(state, dict) and "fc.weight" in state:
    resnet_model.load_state_dict(state)
elif not isinstance(state, dict):
    resnet_model = state
else:
    resnet_model.load_state_dict(state)

resnet_model.to(DEVICE_CLS)
resnet_model.eval()
print("ResNet classifier loaded.\n")

# helper funcs
def classify_fresh_rotten(crop_bgr):
    """
    Takes a cropped BGR image (numpy array from OpenCV),
    returns (label_string, confidence_float).
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return "unknown", 0.0

    # Convert BGR (OpenCV) to RGB (PIL/torch)
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)

    img_tensor = cls_transform(pil_img).unsqueeze(0).to(DEVICE_CLS)

    with torch.no_grad():
        outputs = resnet_model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, dim=1)

    label = CLS_IDX_TO_NAME.get(pred.item(), f"class_{pred.item()}")
    confidence = float(conf.item())
    return label, confidence

# load yolo
try:
    print(f"Loading YOLO model from {YOLO_MODEL_PATH} ...")
    yolo_model = YOLO(YOLO_MODEL_PATH, task="detect")
    print("YOLO model loaded.\n")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    print("Make sure ultralytics is installed and the model file path is correct.")
    raise SystemExit

# open web cam
cap = cv2.VideoCapture(WEBCAM_SOURCE)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    raise SystemExit

columns = [
    "frame", "detection_id",
    "x1", "y1", "x2", "y2",
    "confidence", "class_id", "class_name",
    "fresh_rotten", "fresh_rotten_conf",
    "crop_filename"
]
all_detections_df = pd.DataFrame(columns=columns)

print("Starting webcam detection + classification.")
print("Press 'q' to quit.\n")

frame_count = 0
detection_id = 0
prev_time = time.time()

# main loop
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Webcam frame is empty or end of stream.")
        break

    # run YOLO on the frame
    results = yolo_model.predict(
        frame,
        conf=CONFIDENCE_THRESHOLD,
        imgsz=IMAGE_SIZE,
        stream=False,
        verbose=False,
        device=DEVICE_YOLO
    )

    current_frame_detections = []

    if results and results[0].boxes is not None:
        boxes = results[0].boxes.cpu().numpy()

        for i, box in enumerate(boxes.xyxy):
            x1, y1, x2, y2 = map(int, box)
            conf_det = float(boxes.conf[i])
            class_id = int(boxes.cls[i])
            class_name = yolo_model.names[class_id]

            # crop from original frame
            cropped_img = frame[y1:y2, x1:x2]

            # classify fresh vs rotten
            fr_label, fr_conf = classify_fresh_rotten(cropped_img)

            # save crop
            timestamp = int(time.time() * 1000)
            crop_filename = f"frame_{frame_count}_det_{detection_id}_{class_name}_{fr_label}_{timestamp}.jpg"
            crop_path = os.path.join(SAVE_DIR, crop_filename)

            try:
                cv2.imwrite(crop_path, cropped_img)
            except Exception as crop_error:
                print(f"Error saving crop: {crop_error}")
                crop_filename = "ERROR"

            detection_data = {
                "frame": frame_count,
                "detection_id": detection_id,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "confidence": conf_det,
                "class_id": class_id,
                "class_name": class_name,
                "fresh_rotten": fr_label,
                "fresh_rotten_conf": fr_conf,
                "crop_filename": crop_filename
            }
            current_frame_detections.append(detection_data)
            detection_id += 1

    if current_frame_detections:
        frame_df = pd.DataFrame(current_frame_detections, columns=columns)
        all_detections_df = pd.concat([all_detections_df, frame_df], ignore_index=True)

    # base annotated frame from YOLO (boxes + class labels)
    # instead of using YOLO's .plot(), start from the raw frame
    annotated_frame = frame.copy()

    for det in current_frame_detections:
        x1, y1 = det["x1"], det["y1"]
        x2, y2 = det["x2"], det["y2"]
        fruit_name = det["class_name"]

        fr_label = det["fresh_rotten"]          # fresh / rotten
        fr_conf  = det["fresh_rotten_conf"]     # ResNet confidence
        det_conf = det["confidence"]            # YOLO confidence

        # show both confidences: YOLO + ResNet
        label_text = f"{fruit_name} ({det_conf:.2f}) | {fr_label} ({fr_conf:.2f})"

        # draw bounding box
        cv2.rectangle(
            annotated_frame,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )

        # draw label text
        cv2.putText(
            annotated_frame,
            label_text,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # FPS calculation
    current_time = time.time()
    dt = current_time - prev_time
    fps = 1.0 / dt if dt > 0 else 0.0
    prev_time = current_time

    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(
        annotated_frame,
        fps_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # show frame
    cv2.imshow("Rotten Fruit Detection (YOLO + ResNet)", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# cleanup folder
cap.release()
cv2.destroyAllWindows()

print("\n--- Script Finished ---")
print(f"Total frames processed: {frame_count}")
print(f"Total detections logged: {len(all_detections_df)}")
print(f"Cropped images saved to: {os.path.abspath(SAVE_DIR)}")
