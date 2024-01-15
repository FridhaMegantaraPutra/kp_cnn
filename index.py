import os
import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.image import resize
from st_audiorec import st_audiorec
import io

# Load the saved model
model = load_model('klasifikasi_makhrojul_huruf.h5')

# Define the target shape for input spectrograms
target_shape = (128, 128)

# Define your class labels
classes = ['١', 'ع']

# Function to preprocess and classify an audio file


def test_audio(audio_data, model):
    # Convert the BytesIO object to numpy array
    wav_audio_data_np, _ = librosa.load(io.BytesIO(audio_data), sr=44100)

    mel_spectrogram = librosa.feature.melspectrogram(
        y=wav_audio_data_np, sr=44100)
    mel_spectrogram = resize(np.expand_dims(
        mel_spectrogram, axis=-1), target_shape)
    mel_spectrogram = tf.reshape(mel_spectrogram, (1,) + target_shape + (1,))

    # Make predictions
    predictions = model.predict(mel_spectrogram)

    # Get the class probabilities
    class_probabilities = predictions[0]

    # Get the predicted class index
    predicted_class_index = np.argmax(class_probabilities)

    return class_probabilities, predicted_class_index


# Streamlit app
st.title('Tes Ketepatan Makhrojul huruf')

# Sidebar for navigation
st.sidebar.title('MENU')
page = st.sidebar.radio(
    'Go to', ['Tes Ketepatan', 'makhrojul huruf', 'penggunaan aplikasi'])

# Home page
if page == 'Tes Ketepatan':
    # Record audio
    wav_audio_data = st_audiorec()

    if wav_audio_data is not None:
        st.audio(wav_audio_data, format='audio/wav')

        # Test the recorded audio
        class_probabilities, predicted_class_index = test_audio(
            wav_audio_data, model)

        # Display results for all classes
        st.subheader('Class Probabilities:')
        for i, class_label in enumerate(classes):
            probability = class_probabilities[i]
            st.write(f'Class: {class_label}, Probability: {probability:.4f}')

        # Calculate and display the predicted class and accuracy
        predicted_class = classes[predicted_class_index]
        accuracy = class_probabilities[predicted_class_index]
        st.subheader(f'The audio is classified as: {predicted_class}')
        st.subheader(f'Accuracy: {accuracy:.4f}')

# Guide page
elif page == 'makhrojul huruf':
    st.subheader('cara untuk melafalkan huruf hijaiyah  ١  dan ع')

    st.image('1.jpg', caption='gambar makhrojul huruf', use_column_width=True)

    st.write('Dari pangkal tenggorokan untuk huruf أ dan ه')
    st.write('Dari tengah tenggang untuk huruf ح, ع')


elif page == 'penggunaan aplikasi':
    st.subheader('panduan penggunaan')

    st.write('selamat datang di aplikasi pendeteksi ketepatan makhrojul huruf')

    st.write('1. klik tombol record')
    st.write('2. apabilah sudah selesai melafadzkan langsung klik stop audio juka bisa di donwload dan di reset')
    st.write('3. tunggu hingga probabilitas prediksi muncul')
    st.write(
        '4. pengucapan huruf menggunakan kasro,doma,fatha,kasrotain,dhomatain,fathatain')
