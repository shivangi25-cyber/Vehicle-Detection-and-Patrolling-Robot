import cv2
import torch
import easyocr
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
import os

# === Step 1: RTSP Stream Setup ===
rtsp_url = "rtsp://192.168.1.1/live/ch00_1"  # 🔁 Replace with your RTSP stream
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    raise IOError("❌ Unable to open RTSP stream. Check the URL and camera.")

# === Step 2: Load Models ===
model = YOLO("yolov8n.pt")
reader = easyocr.Reader(['en'])

data = []
serial = 1

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame from RTSP stream.")
        break

    results = model(frame)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]

        if class_name in ['car', 'bus', 'truck', 'motorbike']:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped = frame[y1:y2, x1:x2]

            ocr_results = reader.readtext(cropped)

            for bbox, text, conf in ocr_results:
                if conf > 0.3 and len(text.strip()) >= 3:
                    print(f"Detected Plate: {text.strip()} (Confidence: {conf:.2f})")

                    (tl, tr, br, bl) = bbox
                    tl = (int(tl[0]) + x1, int(tl[1]) + y1)
                    br = (int(br[0]) + x1, int(br[1]) + y1)

                    cv2.rectangle(frame, tl, br, (0, 255, 0), 2)
                    cv2.putText(frame, text.strip(), tl, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    data.append([serial, class_name, text.strip(), timestamp])
                    serial += 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, class_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # === Step 3: Show Stream ===
    cv2.imshow("RTSP Vehicle + Plate Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("🛑 Quitting...")
        break

# === Step 4: Cleanup ===
cap.release()
cv2.destroyAllWindows()

# === Step 5: Save Data ===
if data:
    df = pd.DataFrame(data, columns=["Serial No.", "Vehicle Type", "License Plate", "Timestamp"])
    excel_path = "vehicle_detections.xlsx"

    if os.path.exists(excel_path):
        old_df = pd.read_excel(excel_path)
        df = pd.concat([old_df, df], ignore_index=True)

    df.to_excel(excel_path, index=False)
    print(f"✅ Detection data written to '{excel_path}'")
else:
    print("⚠️ No valid license plates detected.")
