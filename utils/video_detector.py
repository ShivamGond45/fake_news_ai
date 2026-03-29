import cv2
import streamlit as st
from utils.deepfake_predict import predict_image
from utils.face_detection import extract_face

def analyze_video(video_path, frame_placeholder):

    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    fake_frames = 0
    checked_frames = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # Resize frame
        frame = cv2.resize(frame,(320,240))

        # Show frame
        frame_placeholder.image(frame, channels="BGR")

        # Sample frames
        if frame_count % 15 != 0:
            continue

        checked_frames += 1

        try:

            face = extract_face(frame)

            result, score = predict_image(face)

            if result == "FAKE":
                fake_frames += 1

        except:
            continue

    cap.release()

    if checked_frames == 0:
        return "NO FACE DETECTED",0

    fake_ratio = fake_frames / checked_frames

    if fake_ratio > 0.45:
        return "🚨 FAKE VIDEO",fake_ratio
    else:
        return "✅ REAL VIDEO",1-fake_ratio