import os

# Force Streamlit to use a writable .streamlit folder in the current directory
os.environ["XDG_CONFIG_HOME"] = os.getcwd()



import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import tempfile

# Load model
model_path = "deepmedscan_model.h5"
model = tf.keras.models.load_model(model_path)

# Force model to build
_ = model.predict(np.zeros((1, 150, 150, 3)))

# Get last Conv2D layer name
conv_layer_names = [layer.name for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
if not conv_layer_names:
    st.error("No Conv2D layers found for Grad-CAM.")
    st.stop()

last_conv_layer_name = conv_layer_names[-1]

# Grad-CAM Function
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_channel = predictions[:, 0]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap

# Streamlit UI
st.set_page_config(page_title="DeepMedScan", layout="centered")
st.title("🩻 DeepMedScan: Chest X-ray Classifier")
st.write("Upload a chest X-ray to detect **Pneumonia** or confirm it's **Normal**.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    img_resized = image.resize((150, 150))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    label = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    if label == "PNEUMONIA":
        st.error(f"⚠️ Prediction: {label} ({confidence:.2%} confidence)")
    else:
        st.success(f"✅ Prediction: {label} ({confidence:.2%} confidence)")

    st.subheader("📊 Grad-CAM Explanation")
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

    img = np.array(img_resized)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    cv2.imwrite(tmp_file.name, superimposed_img)
    st.image(tmp_file.name, caption="Grad-CAM Heatmap", use_column_width=True)
    tmp_file.close()
    os.remove(tmp_file.name)
