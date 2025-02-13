import streamlit as st
import numpy as np
import cv2
from io import BytesIO
import base64
from streamlit_paste_button import paste_image_button as pbutton  # Updated import

# Colorblindness Simulation Matrices
CVD_MATRICES = {
    "Deuteranopia (Red-Green)": np.array([[0.43, 0.72, -0.15], [0.34, 0.57, 0.09], [-0.02, 0.03, 1.0]]),
    "Protanopia (Red-Green)": np.array([[0.20, 0.99, -0.19], [0.16, 0.79, 0.04], [0.01, -0.02, 1.0]]),
    "Tritanopia (Blue-Yellow)": np.array([[0.97, 0.11, -0.08], [0.02, 0.82, 0.16], [-0.06, 0.88, 0.18]])
}

def apply_colorblind_filter(image, matrix):
    """Applies a colorblindness simulation matrix to an image."""
    image = image.astype(np.float32) / 255.0  # Normalize to range [0,1]
    transformed = np.dot(image, matrix.T)
    transformed = np.clip(transformed, 0, 1)  # Ensure valid RGB range
    return (transformed * 255).astype(np.uint8)

def get_image_download_link(img, filename="colorblind_simulation.png"):
    """Generates a download link for the processed image."""
    _, buffer = cv2.imencode(".png", img)
    b64 = base64.b64encode(buffer).decode()
    return f'<a href="data:image/png;base64,{b64}" download="{filename}">Download Processed Image</a>'

# Streamlit App Layout
st.title("🎨 CVD-Sim: Colorblindness Simulator")

st.sidebar.header("Upload or Paste an Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# # 📋 Paste from Clipboard
# pasted_image = pbutton(
#     label="📋 Paste Image from Clipboard", key="paste_button",
#     errors="raise" # displays errors as st.error messages.
#     )

# if pasted_image is not None:
#     # uploaded_file = BytesIO(pasted_image)
#     st.write('Pasted image:')
#     st.image(pasted_image.image_data)

if uploaded_file:
    image = np.array(cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR (OpenCV) to RGB

    col1, col2 = st.columns(2)
    col1.image(image, caption="Original Image", use_column_width=True)

    st.sidebar.subheader("Select Colorblindness Type")
    option = st.sidebar.radio("", list(CVD_MATRICES.keys()))

    if option:
        processed_image = apply_colorblind_filter(image, CVD_MATRICES[option])
        col2.image(processed_image, caption=f"{option} Simulation", use_column_width=True)
        
        st.markdown(get_image_download_link(processed_image), unsafe_allow_html=True)
