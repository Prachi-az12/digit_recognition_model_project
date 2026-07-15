# # # pip install opencv-python
# # # pip install streamlit-drawable-canvas

# # import streamlit as st
# # import numpy as np
# # import pandas as pd
# # import cv2
# # from streamlit_drawable_canvas import st_canvas
# # from tensorflow.keras.models import load_model

# # # Load the pre-trained models
# # #model = load_model('digit_recognition_model.keras')

# # model = load_model("digit_recognition_model.h5")
# #   # HDF5 format


# # st.title("Handwritten Digit Recognition")

# # canvas_result = st_canvas(
# #     fill_color = "#00000000",  # Canvas background color -> black
# #     stroke_width = 10,
# #     stroke_color ="#FFFFFF",  # Stroke color -> white
# #     background_color ="#FFFFFFF",
# #     width = 280,
# #     height = 280,
# #     drawing_mode = "freedraw",
# #     key = "canvas",
# # )
# # if st.button("Predict"):
# #     st.write("Predicting...")

# #     # Convert the canvas image to a numpy array
# #     img = canvas_result.image_data.astype(np.uint8)

# #     #Convert image to greyscale
# #     grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# #     # Resize the image to 28x28 pixels
# #     grey_img = cv2.resize(grey_img, (28, 28))    

# #     # Normalize the pixel values to be between 0 and 1
# #     grey_img = grey_img / 255.0

# #     # Reshape the image to match the input shape of the model
# #     grey_img = grey_img.reshape(-1,784)

# #     result = model.predict(grey_img)    # Predict the digit using the pre-trained model

# #     index = np.argmax(result)   # Get the index of the highest probability digit

# #     st.write(f"The predicted digit is: {index}")









# pip install opencv-python-headless
# pip install streamlit-drawable-canvas
# pip install tensorflow-cpu==2.15.0

import streamlit as st
import numpy as np
import cv2
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model

# Load the pre-trained ANN model (saved with input shape 784)
model = load_model("digit_recognition_model.h5")

st.title("Handwritten Digit Recognition (ANN)")

# Canvas for drawing digits
canvas_result = st_canvas(
    fill_color="#00000000",   # Transparent background
    stroke_width=10,
    stroke_color="#FFFFFF",   # White stroke
    background_color="#000000",  # Black background
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Predict"):
    st.write("Predicting...")

    # Convert canvas to numpy array
    img = canvas_result.image_data.astype(np.uint8)

    # Convert to grayscale
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize to 28x28
    grey_img = cv2.resize(grey_img, (28, 28))

    # Normalize
    grey_img = grey_img / 255.0

    # Flatten for ANN (1 sample, 784 features)
    grey_img = grey_img.reshape(1, 784)

    # Predict
    result = model.predict(grey_img)
    index = np.argmax(result)

    st.write(f"The predicted digit is: {index}")
