import cv2
import os

def extract_frames(video_folder, save_folder):

    os.makedirs(save_folder, exist_ok=True)

    for video in os.listdir(video_folder):

        path = os.path.join(video_folder, video)
        cap = cv2.VideoCapture(path)

        count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if count % 10 == 0:   # every 10th frame
                filename = os.path.join(save_folder, f"{video}_{count}.jpg")
                cv2.imwrite(filename, frame)

            count += 1

        cap.release()

# RUN
extract_frames("dataset/real_videos", "dataset/frames/real")
extract_frames("dataset/fake_videos", "dataset/frames/fake")