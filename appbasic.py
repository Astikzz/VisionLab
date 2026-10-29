"""
VisionLab - app.py
Streamlit website for the VisionLab project.

This file only builds the web page. All image processing is done by the
functions in image_processing.py (written by Astik) - we just import them.
"""

import cv2
import numpy as np
import streamlit as st

# ---------------------------------------------------------------
# 1. Import Astik's functions (image_processing.py must be in the
#    SAME folder as this file)
# ---------------------------------------------------------------
try:
    from image_processing import (
        to_grayscale,
        gaussian_blur,
        median_filter,
        canny_edges,
        otsu_threshold,
        histogram_equalization,
    )
except ImportError as error:
    st.set_page_config(page_title="VisionLab")
    st.error(
        "Could not import image_processing.py. Make sure it is in the same "
        f"folder as app.py. Details: {error}"
    )
    st.stop()

# ---------------------------------------------------------------
# 2. Page settings
# ---------------------------------------------------------------
st.set_page_config(page_title="VisionLab", page_icon="🔬", layout="wide")

# Dropdown text -> function to call
OPERATIONS = {
    "Grayscale": to_grayscale,
    "Gaussian Blur": gaussian_blur,
    "Median Filter": median_filter,
    "Canny Edge Detection": canny_edges,
    "Otsu Threshold": otsu_threshold,
    "Histogram Equalization": histogram_equalization,
}


# ---------------------------------------------------------------
# 3. Small helper functions
# ---------------------------------------------------------------
def load_image(uploaded_file):
    """Read the uploaded file with OpenCV. Returns a BGR array,
    or None if the file is not a valid image."""
    file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


def show_image(img, caption):
    """Show an image in Streamlit.
    - 2D (black and white) results are shown with clamp=True
    - Colour images are converted BGR -> RGB first (Streamlit expects RGB)"""
    if img.ndim == 2:
        st.image(img, caption=caption, clamp=True)
    else:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=caption)


# ---------------------------------------------------------------
# 4. Title and description
# ---------------------------------------------------------------
st.title("🔬 VisionLab")
st.subheader("Interactive Image Enhancement & Analysis Studio")
st.write(
    "Upload an image, choose an operation (grayscale, blur, noise removal, "
    "edge detection, thresholding or contrast improvement), click **Apply**, "
    "and compare the original with the result. You can then download the "
    "processed image as a PNG."
)

# "Memory" that survives Streamlit re-runs (needed so the result does not
# disappear when you click the Download button)
if "result" not in st.session_state:
    st.session_state.result = None
if "file_id" not in st.session_state:
    st.session_state.file_id = None

# ---------------------------------------------------------------
# 5. Upload
# ---------------------------------------------------------------
uploaded = st.file_uploader(
    "Upload an image (JPG, JPEG or PNG)", type=["jpg", "jpeg", "png"]
)

if uploaded is None:
    st.session_state.result = None
    st.session_state.file_id = None
    st.info("Please upload an image to begin.")
    st.stop()

# If the user uploaded a different file, forget the old result
current_id = f"{uploaded.name}-{uploaded.size}"
if st.session_state.file_id != current_id:
    st.session_state.file_id = current_id
    st.session_state.result = None

# Decode the image (BGR) with error handling
try:
    original = load_image(uploaded)
except Exception as error:
    st.error(f"Could not read the uploaded file: {error}")
    st.stop()

if original is None:
    st.error("This file is not a valid image. Please upload a real JPG or PNG.")
    st.stop()

# ---------------------------------------------------------------
# 6. Choose operation + Apply button
# ---------------------------------------------------------------
choice = st.selectbox("Choose an operation", list(OPERATIONS.keys()))
apply_clicked = st.button("Apply", type="primary")

if apply_clicked:
    try:
        with st.spinner("Processing..."):
            processed = OPERATIONS[choice](original)  # original is not modified
            # Encode as PNG (the image is still in BGR, which OpenCV expects)
            success, buffer = cv2.imencode(".png", processed)
            if not success:
                raise RuntimeError("Could not encode the result as a PNG.")
        st.session_state.result = {
            "image": processed,
            "png_bytes": buffer.tobytes(),
            "operation": choice,
        }
    except Exception as error:
        st.session_state.result = None
        st.error(f"Processing failed: {error}")

# ---------------------------------------------------------------
# 7. Show original and processed side by side
# ---------------------------------------------------------------
left, right = st.columns(2)

with left:
    st.markdown("### Original")
    show_image(original, "Original image")

with right:
    st.markdown("### Processed")
    result = st.session_state.result
    if result is None:
        st.info("Choose an operation and click Apply to see the result here.")
    else:
        show_image(result["image"], f"Result: {result['operation']}")
        file_name = result["operation"].lower().replace(" ", "_") + ".png"
        st.download_button(
            label="Download processed image (PNG)",
            data=result["png_bytes"],
            file_name=file_name,
            mime="image/png",
        )
