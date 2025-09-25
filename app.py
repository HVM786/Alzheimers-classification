import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import tempfile
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# === Updated paths for all models ===
MODEL_PATHS = {
    "VGG16": r"C:\Alzheimer project\model_outputs\vgg16_stream_kfoldfinetuned\fold_5\finetuned_model.keras",
    "ResNet50": r"C:\Alzheimer project\model_outputs\resnet50_stream_kfold_finetuned\fold_5\finetuned_model.keras",
    "DenseNet121": r"C:\Alzheimer project\model_outputs\densenet121_final\best_model.keras",
    "EfficientNetV2S": r"C:\Alzheimer project\model_outputs\efficientnetv2s_final\best_model.keras"
}

# === Lazy load model cache ===
_loaded_models = {}
def get_model(model_name):
    if model_name not in _loaded_models:
        st.write(f"Loading {model_name} model…")
        _loaded_models[model_name] = load_model(MODEL_PATHS[model_name])
    return _loaded_models[model_name]

# === Class labels ===
label_map = {0: "Mild Dementia", 1: "Moderate Dementia", 2: "Non Demented"}
class_names = list(label_map.values())

# === VizGradCAM logic ===
def vizgradcam_heatmap(model, img_array, pred_index=None):
    preds = model.predict(img_array)
    pred_index = np.argmax(preds[0]) if pred_index is None else pred_index
    
    # Find last Conv2D layer automatically
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer.name
            break
    if last_conv_layer is None:
        raise ValueError("No Conv2D layer found in model.")

    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_output = predictions[:, pred_index]

    grads = tape.gradient(class_output, conv_outputs)
    if grads is None:
        heatmap = np.zeros((224, 224), dtype=np.float32)
    else:
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()
        if heatmap.ndim != 2 or np.isnan(heatmap).any():
            heatmap = np.zeros((224, 224), dtype=np.float32)
        else:
            heatmap = cv2.resize(heatmap.astype(np.float32), (224, 224), interpolation=cv2.INTER_LINEAR)

    return heatmap, preds[0]

def overlay_heatmap(original_img, heatmap, alpha=0.4):
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_HOT)
    return cv2.addWeighted(original_img, 1 - alpha, heatmap_colored, alpha, 0)

# === Streamlit GUI ===
st.title(" Alzheimer’s MRI Classifier with VizGradCAM Explainability")
st.write("Upload an MRI slice, select a model or Ensemble, and see predictions with averaged Grad-CAM heatmaps.")

# Model selector
model_choice = st.selectbox(
    "Choose Model:",
    ["VGG16", "ResNet50", "DenseNet121", "EfficientNetV2S", "Ensemble (All 4)"]
)

uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    # Preprocess image
    img = image.load_img(tfile.name, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array / 255.0, axis=0)
    original_bgr = cv2.cvtColor(np.uint8(img_array[0] * 255), cv2.COLOR_RGB2BGR)

    if model_choice.startswith("Ensemble"):
        # Run all 4 models → collect predictions + heatmaps
        all_preds, all_heatmaps = [], []
        for m_name in ["VGG16", "ResNet50", "DenseNet121", "EfficientNetV2S"]:
            mdl = get_model(m_name)
            heatmap, preds = vizgradcam_heatmap(mdl, img_array)
            all_preds.append(preds)
            all_heatmaps.append(heatmap)
        # Average predictions + heatmaps
        avg_preds = np.mean(np.stack(all_preds, axis=0), axis=0)
        avg_heatmap = np.mean(np.stack(all_heatmaps, axis=0), axis=0)
        
        pred_index = np.argmax(avg_preds)
        pred_label = label_map[pred_index]
        conf = avg_preds[pred_index]
        final_overlay = overlay_heatmap(original_bgr, avg_heatmap)
        probs = avg_preds

    else:
        # Single model prediction
        mdl = get_model(model_choice)
        heatmap, preds = vizgradcam_heatmap(mdl, img_array)
        pred_index = np.argmax(preds)
        pred_label = label_map[pred_index]
        conf = preds[pred_index]
        final_overlay = overlay_heatmap(original_bgr, heatmap)
        probs = preds

    # Show side-by-side: Original + Overlay
    side_by_side = np.hstack((original_bgr, final_overlay))
    st.image(side_by_side, caption=f"Prediction: {pred_label} | Confidence: {conf*100:.2f}%", use_column_width=True)
    st.success(f"**Predicted Class:** {pred_label}")
    st.info(f"**Confidence Score:** {conf*100:.2f}%")

    # Show bar chart for all class probabilities
    st.write("### Prediction Probabilities")
    fig, ax = plt.subplots()
    ax.bar(class_names, probs * 100, color=['#FFA07A', '#FF4500', '#90EE90'])
    ax.set_ylabel("Confidence (%)")
    ax.set_ylim(0, 100)
    for i, v in enumerate(probs * 100):
        ax.text(i, v + 1, f"{v:.1f}%", ha='center', fontweight='bold')
    st.pyplot(fig)

    if model_choice.startswith("Ensemble"):
        st.warning("Ensemble Grad-CAM = averaged heatmaps from all 4 models.")

