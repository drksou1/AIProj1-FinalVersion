""" For the live demonstration, we will use the webcam to capture video frames and run our trained model on them in real-time. 
This script will load the trained model, access the webcam, and display the video feed with predictions overlaid on it. """

import cv2
import torch
from torchvision.transforms.functional import to_tensor

from args import DEVICE
from model import build_model

MODEL_PATH = "runs/best_model.pth"
SCORE_THRESHOLD = 0.5

device = torch.device(DEVICE)

model = build_model("fasterrcnn_resnet50_fpn", num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

print("Webcam started. Press q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # OpenCV uses BGR, PyTorch expects RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_tensor = to_tensor(rgb_frame).to(device)

    with torch.no_grad():
        prediction = model([image_tensor])[0]

    boxes = prediction["boxes"].cpu()
    scores = prediction["scores"].cpu()

    for box, score in zip(boxes, scores):
        if score < SCORE_THRESHOLD:
            continue

        x1, y1, x2, y2 = box.int().tolist()

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            frame,
            f"chair {score:.2f}",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    cv2.imshow("Chair detection demo", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()