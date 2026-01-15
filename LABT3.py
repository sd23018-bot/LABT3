import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import requests


st.set_page_config(
    page_title="Real-Time Webcam Image Classification",
    layout="centered"
)

st.title("Real-Time Webcam Image Classification (ResNet-18)")
st.write("Capture an image using your webcam and classify it using a pretrained ResNet-18 model.")


device = torch.device("cpu")
st.write(f"Running on device: {device}")


LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
labels = requests.get(LABELS_URL).text.splitlines()


model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()
model.to(device)


preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


camera_image = st.camera_input("Capture image from webcam")

if camera_image is not None:
  
    image = Image.open(camera_image).convert("RGB")
    st.image(image, caption="Captured Image", use_container_width=True)

    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(device)

  
    with torch.no_grad():
        output = model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

   
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    results = []
    for i in range(5):
        results.append({
            "Label": labels[top5_catid[i]],
            "Probability (%)": float(top5_prob[i] * 100)
        })

   
    st.subheader("Top 5 Predictions")
    df = pd.DataFrame(results)
    st.table(df)

    # -------------------- Visualization --------------------
    st.subheader("Prediction Confidence")
    st.bar_chart(df.set_index("Label"))
