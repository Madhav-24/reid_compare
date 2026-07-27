import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import torchreid

# --------------------------------------------------
# Configuration
# --------------------------------------------------
IMG1 = "img1.jpg"
IMG2 = "img2.jpg"
OUTPUT = "comparison_result.jpg"

MATCH_THRESHOLD = 0.75  # Adjust if needed

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------
# Load pretrained ReID model
# --------------------------------------------------
model = torchreid.models.build_model(
    name="osnet_x1_0",
    num_classes=1000,
    pretrained=True
)

model.to(DEVICE)
model.eval()

# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------
transform = T.Compose([
    T.Resize((256, 128)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def extract_feature(img_path):
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        feat = model(tensor)

    feat = F.normalize(feat, dim=1)
    return feat.cpu().numpy()[0]


# --------------------------------------------------
# Extract Features
# --------------------------------------------------
feat1 = extract_feature(IMG1)
feat2 = extract_feature(IMG2)

# --------------------------------------------------
# Cosine Similarity
# --------------------------------------------------
similarity = np.dot(feat1, feat2)

match = similarity > MATCH_THRESHOLD

# --------------------------------------------------
# Load images
# --------------------------------------------------
img1 = cv2.imread(IMG1)
img2 = cv2.imread(IMG2)

h = max(img1.shape[0], img2.shape[0])

img1 = cv2.resize(
    img1,
    (int(img1.shape[1] * h / img1.shape[0]), h)
)

img2 = cv2.resize(
    img2,
    (int(img2.shape[1] * h / img2.shape[0]), h)
)

combined = np.hstack([img1, img2])

# --------------------------------------------------
# Create canvas
# --------------------------------------------------
top_space = 120
bottom_space = 220

canvas = np.ones(
    (
        combined.shape[0] + top_space + bottom_space,
        combined.shape[1],
        3
    ),
    dtype=np.uint8
) * 255

canvas[top_space:top_space+combined.shape[0]] = combined

# --------------------------------------------------
# Title
# --------------------------------------------------
if match:
    color = (0, 180, 0)
    title = "OK (MATCH)"
else:
    color = (0, 0, 255)
    title = "NOT EQUAL"

cv2.putText(
    canvas,
    title,
    (20, 60),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.6,
    color,
    3
)

cv2.putText(
    canvas,
    f"Similarity: {similarity:.4f}",
    (20, 100),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0,0,0),
    2
)

# --------------------------------------------------
# Display feature vectors
# --------------------------------------------------
text1 = "Img1: " + np.array2string(
    feat1[:10],
    precision=3,
    separator=", "
)

text2 = "Img2: " + np.array2string(
    feat2[:10],
    precision=3,
    separator=", "
)

y = top_space + combined.shape[0] + 35

cv2.putText(
    canvas,
    text1,
    (10, y),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.45,
    (0,0,0),
    1
)

cv2.putText(
    canvas,
    text2,
    (10, y + 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.45,
    (0,0,0),
    1
)

cv2.putText(
    canvas,
    f"Feature Dimension = {len(feat1)}",
    (10, y + 70),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (150,0,0),
    2
)

# --------------------------------------------------
# Save
# --------------------------------------------------
cv2.imwrite(OUTPUT, canvas)

print("="*50)
print("Similarity :", similarity)
print("Result     :", title)
print("Saved      :", OUTPUT)
print("="*50)