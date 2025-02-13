import streamlit as st
import numpy as np
import cv2
from io import BytesIO
import base64
from streamlit_paste_button import paste_image_button as pbutton  # Updated import

# # 📋 Paste from Clipboard
# pasted_image = pbutton(
#     label="📋 Paste Image from Clipboard", key="paste_button",
#     errors="raise" # displays errors as st.error messages.
#     )

# if pasted_image is not None:
#     # uploaded_file = BytesIO(pasted_image)
#     st.write('Pasted image:')
#     st.image(pasted_image.image_data)

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

# Streamlit App Layout
st.header("🎨 CVD Sim: :blue[A Color Vision Deficiency Simulator]")

st.sidebar.header("Upload an Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = np.array(cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR (OpenCV) to RGB

    # Show the original image
    st.image(image, caption="🖼 Original Image", use_container_width=True)

    # Sidebar checkboxes for selecting simulations
    st.sidebar.subheader("Select Colorblindness Simulations")
    selected_simulations = {name: st.sidebar.checkbox(name, False) for name in CVD_MATRICES.keys()}

    # Display selected simulations
    for sim_type, enabled in selected_simulations.items():
        if enabled:
            processed_image = apply_colorblind_filter(image, CVD_MATRICES[sim_type])
            st.image(processed_image, caption=f"🎨 {sim_type} Simulation", use_container_width =True)