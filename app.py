"""
VisionLab PRO - extra features on top of the basic app.
Run:  streamlit run app.py

Extra features:
  1. Smart Assistant  - analyses the image (brightness, contrast, noise) and
                        recommends operations (rule-based, no AI model needed)
  2. Filter chain     - apply several operations one after another
  3. Histogram        - brightness histogram of original vs processed
  4. Blend slider     - smoothly fade between original and processed image
"""

import cv2
import numpy as np
import pandas as pd
import streamlit as st

try:
    import image_processing as ip
except ImportError as error:
    st.set_page_config(page_title="VisionLab PRO")
    st.error(f"Could not import image_processing.py (same folder needed): {error}")
    st.stop()

OPERATIONS = {
    "Grayscale": ip.to_grayscale,
    "Gaussian Blur": ip.gaussian_blur,
    "Median Filter": ip.median_filter,
    "Canny Edge Detection": ip.canny_edges,
    "Otsu Threshold": ip.otsu_threshold,
    "Histogram Equalization": ip.histogram_equalization,
}

DESCRIPTIONS = {
    "Grayscale": "Converts the colour image to shades of gray.",
    "Gaussian Blur": "Smooths the image and reduces mild noise.",
    "Median Filter": "Removes salt-and-pepper (impulse) noise while keeping edges.",
    "Canny Edge Detection": "Finds the edges (outlines) of objects.",
    "Otsu Threshold": "Automatically makes a black-and-white image.",
    "Histogram Equalization": "Improves contrast by spreading out brightness values.",
}


# ---------------- helper functions ----------------
def to_display(img):
    """OpenCV is BGR, Streamlit needs RGB for colour images."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img


def run_pipeline(image, ops, params):
    """Apply the chosen operations one after another."""
    out = image
    for name in ops:
        if name in ("Gaussian Blur", "Median Filter"):
            out = OPERATIONS[name](out, kernel_size=params["kernel"])
        elif name == "Canny Edge Detection":
            out = OPERATIONS[name](out, params["low"], params["high"])
        else:
            out = OPERATIONS[name](out)
    return out


def analyze(image):
    """Measure simple image statistics for the Smart Assistant."""
    gray = ip.to_grayscale(image)
    h, w = gray.shape

    brightness = float(gray.mean())
    contrast = float(gray.std())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Salt-and-pepper noise: pixels very different from their median neighbours
    med = cv2.medianBlur(gray, 3)
    impulse = float((cv2.absdiff(gray, med) > 60).mean() * 100)

    # Gaussian noise estimate (Immerkaer's method)
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    conv = cv2.filter2D(gray.astype(np.float64), -1, k)[1:-1, 1:-1]
    sigma = float(np.sqrt(np.pi / 2) * np.abs(conv).sum() / (6.0 * (w - 2) * (h - 2)))

    return {
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness,
        "impulse_noise": impulse,
        "noise_sigma": sigma,
    }


def recommend(stats):
    """Rule-based suggestions: returns (list of (operation, reason), pipeline)."""
    tips, pipeline = [], []
    if stats["impulse_noise"] > 1.5:
        tips.append(("Median Filter",
                     f"{stats['impulse_noise']:.1f}% of pixels look like salt-and-pepper noise."))
        pipeline.append("Median Filter")
    elif stats["noise_sigma"] > 8:
        tips.append(("Gaussian Blur",
                     f"Estimated noise level is high (sigma = {stats['noise_sigma']:.1f})."))
        pipeline.append("Gaussian Blur")
    if stats["contrast"] < 45:
        tips.append(("Histogram Equalization",
                     f"Contrast is low (std = {stats['contrast']:.0f}); the image looks flat."))
        pipeline.append("Histogram Equalization")
    if stats["brightness"] < 60:
        tips.append(("Histogram Equalization", "The image is quite dark."))
        if "Histogram Equalization" not in pipeline:
            pipeline.append("Histogram Equalization")
    if not tips:
        tips.append(("Canny Edge Detection",
                     "The image is clean with good contrast, so it is ready for edge detection."))
        pipeline.append("Canny Edge Detection")
    return tips, pipeline


def blend(original, processed, alpha):
    """alpha = 0 -> original, alpha = 1 -> processed."""
    if processed.ndim == 2:
        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(original, 1 - alpha, processed, alpha, 0)


def histogram_frame(original, processed):
    g1 = ip.to_grayscale(original)
    g2 = ip.to_grayscale(processed)
    h1 = cv2.calcHist([g1], [0], None, [256], [0, 256]).ravel()
    h2 = cv2.calcHist([g2], [0], None, [256], [0, 256]).ravel()
    return pd.DataFrame({"Original": h1, "Processed": h2})


# ===== UI =====
st.set_page_config(page_title="VisionLab PRO", page_icon="🔬", layout="wide")

st.markdown(
    """
    <style>
    .hero {padding: 1.2rem 1.6rem; border-radius: 16px; margin-bottom: 1rem;
           background: linear-gradient(120deg, #4f46e5 0%, #9333ea 50%, #ec4899 100%);}
    .hero h1 {color: white; margin: 0; font-size: 2.2rem;}
    .hero p {color: #f5f3ff; margin: 0.2rem 0 0 0;}
    </style>
    <div class="hero">
      <h1>🔬 VisionLab PRO</h1>
      <p>Interactive Image Enhancement &amp; Analysis Studio</p>
    </div>
    """,
    unsafe_allow_html=True,
)

for key, default in (("result", None), ("file_id", None)):
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------- Sidebar ----------------
st.sidebar.header("⚙️ Controls")
uploaded = st.sidebar.file_uploader("1. Upload an image", type=["jpg", "jpeg", "png"])
ops = st.sidebar.multiselect(
    "2. Choose operation(s) - applied in this order",
    list(OPERATIONS),
    default=["Grayscale"],
)
for name in ops:
    st.sidebar.caption(f"**{name}:** {DESCRIPTIONS[name]}")

params = {"kernel": 5, "low": 100, "high": 200}
if any(o in ("Gaussian Blur", "Median Filter") for o in ops):
    params["kernel"] = st.sidebar.slider("Kernel size (higher = stronger)", 3, 15, 5, step=2)
if "Canny Edge Detection" in ops:
    params["low"] = st.sidebar.slider("Canny low threshold", 0, 255, 100)
    params["high"] = st.sidebar.slider("Canny high threshold", 0, 255, 200)

apply_clicked = st.sidebar.button("3. Apply", type="primary")

if uploaded is None:
    st.session_state.result = None
    st.session_state.file_id = None
    st.info("👈 Upload a JPG, JPEG or PNG image in the sidebar to begin.")
    st.stop()

current_id = f"{uploaded.name}-{uploaded.size}"
if st.session_state.file_id != current_id:
    st.session_state.file_id = current_id
    st.session_state.result = None

try:
    original = cv2.imdecode(np.frombuffer(uploaded.getvalue(), np.uint8), cv2.IMREAD_COLOR)
except Exception as error:
    st.error(f"Could not read the uploaded file: {error}")
    st.stop()
if original is None:
    st.error("This file is not a valid image. Please upload a real JPG or PNG.")
    st.stop()


def store_result(image, label):
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Could not encode the result as a PNG.")
    st.session_state.result = {"image": image, "png": buf.tobytes(), "label": label}


if apply_clicked:
    if not ops:
        st.warning("Please choose at least one operation.")
    elif "Canny Edge Detection" in ops and params["low"] >= params["high"]:
        st.error("Canny low threshold must be smaller than the high threshold.")
    else:
        try:
            with st.spinner("Processing..."):
                store_result(run_pipeline(original, ops, params), " → ".join(ops))
        except Exception as error:
            st.session_state.result = None
            st.error(f"Processing failed: {error}")

height, width = original.shape[:2]
st.write(f"**File:** {uploaded.name}  |  **Size:** {width} x {height} px")

tab_compare, tab_hist, tab_blend, tab_ai = st.tabs(
    ["🖼️ Compare", "📊 Histogram", "🎚️ Blend slider", "🤖 Smart Assistant"]
)
result = st.session_state.result

with tab_compare:
    left, right = st.columns(2)
    with left:
        st.markdown("### Original")
        st.image(to_display(original))
    with right:
        st.markdown("### Processed")
        if result is None:
            st.info("Choose operation(s) and click Apply.")
        else:
            st.image(to_display(result["image"]), clamp=True)
            st.caption(f"Applied: {result['label']}")
            safe = result["label"].lower().replace(" → ", "_then_").replace(" ", "_")
            st.download_button("⬇️ Download processed image (PNG)", data=result["png"],
                               file_name=f"visionlab_{safe}.png", mime="image/png")

with tab_hist:
    if result is None:
        st.info("Apply an operation to compare brightness histograms.")
    else:
        st.write("Brightness histogram (0 = black, 255 = white).")
        st.line_chart(histogram_frame(original, result["image"]))

with tab_blend:
    if result is None:
        st.info("Apply an operation to use the blend slider.")
    else:
        alpha = st.slider("Original  ◀────▶  Processed", 0, 100, 100) / 100
        st.image(to_display(blend(original, result["image"], alpha)), clamp=True)

with tab_ai:
    st.write("The assistant measures your image and suggests what to do. "
             "It uses simple rules on image statistics (no AI model).")
    stats = analyze(original)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Brightness", f"{stats['brightness']:.0f} / 255")
    c2.metric("Contrast", f"{stats['contrast']:.0f}")
    c3.metric("Impulse noise", f"{stats['impulse_noise']:.1f}%")
    c4.metric("Noise level", f"{stats['noise_sigma']:.1f}")

    tips, pipeline = recommend(stats)
    st.markdown("#### Recommendations")
    for name, reason in tips:
        st.write(f"✅ **{name}** - {reason}")
    st.write("Suggested chain: " + " → ".join(pipeline))
    if st.button("✨ Apply smart chain"):
        try:
            store_result(run_pipeline(original, pipeline, params), " → ".join(pipeline))
            st.rerun()
        except Exception as error:
            st.error(f"Processing failed: {error}")
