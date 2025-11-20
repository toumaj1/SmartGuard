import cv2
from ultralytics import YOLO

# Charger un modèle YOLO pré-entraîné (tu peux changer "yolov8n.pt" si tu veux)
model = YOLO("yolov8n.pt")


# ----------------------------------------------------
# 🔍 Détection d’image
# ----------------------------------------------------
def detect_image(image_path):
    results = model(image_path)
    boxes_image = cv2.imread(image_path)

    detections = []

    for r in results:
        if hasattr(r, "boxes"):
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                conf = float(box.conf[0])

                if conf >= 0.5:
                    detections.append({
                        "class": cls_name,
                        "confidence": round(conf, 2)
                    })

                    # Dessin des boîtes
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(boxes_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(boxes_image, f"{cls_name} {conf:.2f}",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 0), 2)
    return detections, boxes_image


# ----------------------------------------------------
# 🎥 Détection vidéo
# ----------------------------------------------------
def detect_video(video_path):
    cap = cv2.VideoCapture(video_path)
    events = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        for r in results:
            if hasattr(r, "boxes"):
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = model.names[cls_id]
                    conf = float(box.conf[0])

                    if conf >= 0.5:
                        events.append({"class": cls_name, "confidence": round(conf, 2)})

    cap.release()
    return events


# ----------------------------------------------------
# 🧠 Rapport généré par LLM (version simple sans API)
# ----------------------------------------------------
def generate_report(detections):
    if not detections:
        return "Aucune menace détectée."

    text_report = "Menaces détectées :\n"

    for d in detections:
        text_report += f"- {d['class']} (confiance : {d['confidence']})\n"

    return text_report
