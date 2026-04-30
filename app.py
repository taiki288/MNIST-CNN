import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('./models/mnist_cnn_2.keras')

model = load_model()

st.title("MNIST手書き文字認識アプリ")
st.write("左のキャンバスに 0〜9 の数字を書いてください。")

canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",
    stroke_width=20,
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("判定する"):
    if canvas_result.image_data is not None:
        img = Image.fromarray(canvas_result.image_data.astype(np.uint8)).convert("L")
        
        bbox = img.getbbox()

        if bbox:
            img = img.crop(bbox)
            img = img.resize((28, 28), Image.Resampling.LANCZOS)
            st.write("前処理された画像")
            st.image(img, width=280)
            img_array = np.array(img).astype('float32') / 255.0
            img_array = img_array.reshape(1, 28, 28, 1)
        else:
            st.warning("数字を書いてから判定ボタンを押してください。")

        prediction = model.predict(img_array)
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction)

        st.header(f"予測結果: {predicted_digit}")
        st.write(f"確信度: {confidence:.2%}")
        
        st.bar_chart(prediction[0])
    else:
        st.warning("数字を書いてから判定ボタンを押してください。")