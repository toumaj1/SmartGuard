import streamlit as st
import cv2
import os
import tempfile
from detect_functions import detect_image, detect_video, generate_report

st.title("SmartGuard - Détection Danger")

# -------- Upload d’image --------
uploaded_img = st.file_uploader("Upload une image", type=["jpg","png"])

if uploaded_img:
    # 👉 Fichier temporaire compatible Streamlit Cloud
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_img.getbuffer())
        tmp_path = tmp.name

    detections, img_with_boxes = detect_image(tmp_path)
    st.image(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
    st.write("Détections:", detections)

    report = generate_report(detections)
    st.write("Rapport LLM:", report)

# -------- Upload vidéo --------
uploaded_vid = st.file_uploader("Upload une vidéo", type=["mp4"])

if uploaded_vid:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_vid.getbuffer())
        tmp_vid_path = tmp.name

    events = detect_video(tmp_vid_path)
    st.video(tmp_vid_path)
    st.write("Détections vidéo:", events)
