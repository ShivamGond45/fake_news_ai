import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
from torch import nn

# =========================
# FACE DETECTION
# =========================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def extract_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        return frame[y:y+h, x:x+w]
    else:
        return frame


# =========================
# LOAD MODEL
# =========================
model = models.resnet18()
model.fc = nn.Linear(512, 2)

model.load_state_dict(torch.load("models/video_model.pth", map_location="cpu"))
model.eval()

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3)
])


# =========================
# STEP 2 → PREDICT FRAME
# =========================
def predict_frame(frame):

    # Face part
    face = extract_face(frame)

    face_img = Image.fromarray(face)
    face_img = transform(face_img).unsqueeze(0)

    # Full frame
    full_img = Image.fromarray(frame)
    full_img = transform(full_img).unsqueeze(0)

    with torch.no_grad():
        out_face = model(face_img)
        out_full = model(full_img)

    # Combine outputs
    final_output = (out_face + out_full) / 2

    _, pred = torch.max(final_output, 1)

    return pred.item()


# =========================
# STEP 3 → VIDEO ANALYSIS
# =========================
def analyze_video(video_path, frame_placeholder=None):

    cap = cv2.VideoCapture(video_path)

    fake_count = 0
    real_count = 0
    frame_no = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # हर 10th frame check
        if frame_no % 10 == 0:

            pred = predict_frame(frame)

            # UI display (optional)
            if frame_placeholder is not None:
                frame_placeholder.image(frame, channels="BGR", width=300)

            if pred == 0:
                fake_count += 1
            else:
                real_count += 1

        frame_no += 1

    cap.release()

    total = fake_count + real_count

    if total == 0:
        return "No frames detected", 0

    if fake_count > real_count:
        return "🚨 FAKE VIDEO", fake_count / total
    else:
        return "✅ REAL VIDEO", real_count / total