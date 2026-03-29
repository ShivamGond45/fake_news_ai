import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_autorefresh import st_autorefresh

from utils.deepfake_predict import predict_image
from utils.video_detector import analyze_video
from utils.text_detector import detect_fake_news
from utils.fact_check_api import fact_check_news
from utils.heatmap import generate_heatmap


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Fake News & Deepfake Detector",
    layout="wide",
    page_icon="🤖"
)

# =========================
# TITLE
# =========================
st.markdown(
    "<h1 style='text-align:center;'>🤖 AI Fake News & Deepfake Detector</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<h4 style='text-align:center;'>Detect Fake News, Deepfake Images and Videos using AI</h4>",
    unsafe_allow_html=True
)

st.markdown("---")

# =========================
# SIDEBAR
# =========================
st.sidebar.title("Navigation")

option = st.sidebar.radio(
    "Select Detection Type",
    ["Home","Image Deepfake","Video Deepfake","Fake News Text"]
)

# =========================
# HOME PAGE
# =========================
if option == "Home":

    col1, col2 = st.columns([1,1])

    # LEFT SIDE
    with col1:
        st.markdown("""
        ## 🔍 Features

        - 🖼 Deepfake Image Detection  
        - 🎥 Video Deepfake Detection  
        - 📰 Fake News Text Detection  
        - 🧠 AI Multimodal Analysis  
        """)

    # RIGHT SIDE (SLIDER)
    with col2:

        images = [
            "assets/images/image1.png",
            "assets/images/image2.png",
            "assets/images/image3.png",
            "assets/images/image4.png",
            "assets/images/image5.png"

        ]

        count = st_autorefresh(interval=2000, key="slider")

        index = count % len(images)

        st.image(images[index], use_container_width=True)

    # BELOW
    st.markdown("---")

    st.markdown("## 🚀 AI Detection Modules")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/4712/4712109.png", width=120)
        st.markdown("### Image Deepfake Detection")

    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/3039/3039389.png", width=120)
        st.markdown("### Video Deepfake Detection")

    with col3:
        st.image("https://cdn-icons-png.flaticon.com/512/2965/2965879.png", width=120)
        st.markdown("### Fake News Detection")


# =========================
# IMAGE DEEPFAKE + HEATMAP
# =========================
elif option == "Image Deepfake":

    st.subheader("🖼 Upload Image")

    file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if file:

        image = Image.open(file)
        img = np.array(image)

        st.info("🔍 AI is analyzing image...")

        # Prediction
        result, score = predict_image(img)

        # Heatmap
        heatmap = generate_heatmap(img)

        # DISPLAY
        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(image, caption="Original Image", use_container_width=True)

        with col2:
            st.image(heatmap, caption="🔥 AI Heatmap", use_container_width=True)

        with col3:

            if result == "FAKE":
                st.error("🚨 Fake Image")
            else:
                st.success("✅ Real Image")

            st.progress(score)
            st.write("Confidence:", round(score*100,2), "%")


# =========================
# VIDEO DEEPFAKE
# =========================
elif option == "Video Deepfake":

    st.subheader("🎥 Upload Video")

    video = st.file_uploader("Upload Video", type=["mp4","mov","avi"])

    if video:

        with open("temp_video.mp4","wb") as f:
            f.write(video.read())

        col1, col2 = st.columns([2,1])

        with col1:
            st.video("temp_video.mp4")

        with col2:

            st.info("🤖 AI scanning frames...")

            frame_placeholder = st.empty()

            result, confidence = analyze_video("temp_video.mp4", frame_placeholder)

            if "FAKE" in result:
                st.error(result)
            else:
                st.success(result)

            st.progress(confidence)
            st.write("Confidence:", round(confidence*100,2), "%")


# =========================
# FAKE NEWS + API
# =========================
elif option == "Fake News Text":

    st.subheader("📰 Fake News Detection")

    text = st.text_area("Paste News Text")

    if st.button("Analyze News"):

        st.info("🔍 AI + API analyzing...")

        label, score = detect_fake_news(text)
        api_result = fact_check_news(text)

        if "FAKE" in label.upper():
            st.error("🚨 Fake News Detected (AI Model)")
        else:
            st.success("✅ Real News (AI Model)")

        st.progress(score)
        st.write("Confidence:", round(score*100,2), "%")

        st.markdown("---")

        st.subheader("🌐 Fact Check Result")
        st.info(api_result)