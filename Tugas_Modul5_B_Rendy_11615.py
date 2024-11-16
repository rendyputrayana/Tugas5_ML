import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image

model_path = r'C:\Users\ASUS\Downloads\Introduction to Deep Learning (Praktek)\Introduction to Deep Learning (Praktek)\model_tf.h5'

if os.path.exists(model_path):
    try:
        tf.get_logger().setLevel('ERROR')
        model = tf.keras.models.load_model(model_path, compile=False)

        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        
        def preprocess_image(image):
            image = image.resize((28, 28))
            image = image.convert('L')
            image_array = np.array(image) / 255.0
            image_array = image_array.reshape(1, 28, 28, 1)
            return image_array
        
        st.title('Fashion MNIST Image Classification')
        st.write("Unggah gambar item fashion (misal kaos, celana, sepatu, tas, dll) dan model akan mengklasifikasikannya.")

        uploaded_image = st.file_uploader("Pilih gambar", type=['jpg', 'jpeg', 'png'])

        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption='Gambar yang diunggah', use_column_width=True)
            
            if st.button('Predict'):
                preprocessed_image = preprocess_image(image)
                prediction = model.predict(preprocessed_image)[0]

                predicted_class = np.argmax(prediction)
                confidence = prediction[predicted_class] * 100

                st.write("### Hasil Prediksi")
                st.write(f"Kelas Prediksi: **{class_names[predicted_class]}**")
                st.write(f"Confidence: **{confidence:.2f}%**")
    except Exception as e:
        st.write("Error: ", str(e))
else:
    st.write("Model tidak ditemukan")
