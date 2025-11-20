# ====== SmartGuard - app.py ======

import streamlit as st
from ultralytics import YOLO
import cv2
from matplotlib import pyplot as plt
import uuid
import os
from openai import OpenAI

# ---------------- YOLO ----------------
model = YOLO("yolov8n.pt")  # Assure-toi que yolov8n.pt est téléchargé

# ---------------- Fonctions ----------------

def detect_image(image_path):
    """Détecte les objets sur une image et retourne l'image annotée et la liste des détections"""
    results = model(image_path)
    res = results[0]
    img_with_boxes = res.plot()
    detections = []
    if hasattr(res, "boxes"):
        for box in res.boxes:
            cls = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            if conf > 0.5:
                detections.append({"class": cls, "confidence": conf})
    return detections, img_with_boxes

def detect_video(video_path, frame_interval_sec=0.5):
    """Analyse une vidéo en détectant les objets à intervalle de temps défini"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    interval_frames = max(1, int(round(fps * frame_interval_sec)))
    frame_no = 0
    saved_events = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_no % interval_frames == 0:
            tmp_path = f"/content/tmp_{uuid.uuid4().hex}.jpg"
            cv2.imwrite(tmp_path, frame)
            detections, _ = detect_image(tmp_path)
            if detections:
                saved_events.append({"frame": frame_no, "detections": detections})
            os.remove(tmp_path)
        frame_no += 1
    cap.release()
    return saved_events

def generate_report(detections):
    """Génère un rapport LLM à partir des détections"""
    client = OpenAI(api_key="TON_API_KEY")  # Remplace par ta clé OpenAI
    prompt = f"Explique de manière intelligente ce que signifient ces détections : {detections}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message["content"]

# ---------------- Interface Streamlit ----------------

st.title("SmartGuard - Détection Danger")

# Upload Image
uploaded_img = st.file_uploader("Upload une image", type=["jpg","png"])
if uploaded_img:
    tmp_path = f"/content/{uploaded_img.name}"
    with open(tmp_path, "wb") as f:
        f.write(uploaded_img.getbuffer())

    detections, img_with_boxes = detect_image(tmp_path)
    st.image(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
    st.write("Détections:", detections)

    report = generate_report(detections)
    st.write("Rapport LLM:", report)

# Upload Video
uploaded_vid = st.file_uploader("Upload une vidéo", type=["mp4"])
if uploaded_vid:
    tmp_vid_path = f"/content/{uploaded_vid.name}"
    with open(tmp_vid_path, "wb") as f:
        f.write(uploaded_vid.getbuffer())

    events = detect_video(tmp_vid_path)
    st.video(tmp_vid_path)
    st.write("Détections vidéo:", events)
