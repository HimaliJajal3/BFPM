import streamlit as st
from ultralytics import YOLO

st.title("Bone Fracture Prediction")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
# if uploaded:
#     col1, col2 = st.columns(2)

# model.predict(bla bla bla)
# st.image(uploaded)
# st.button("Click here")

model = YOLO("best.pt")

results = model.predict(uploaded)
st.write (results)

if uploaded:
    with col1:
        st.header("Original Image")
        st.image(uploaded)

    with col2:
        st.header("Processed Image")
        st.image("https://static.streamlit.io/examples/dog.jpg")