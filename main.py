import streamlit as st 
from PIL import Image
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np

if 'model' not in st.session_state:
    st.session_state.model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')


def load_image(img):

    img = tf.convert_to_tensor(img, dtype=None, dtype_hint=None, name=None)
    img = tf.image.resize(img, [600,600],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]
    
    img = img.numpy()
    return img


### Excluding Imports ###
st.title("Neural Style Transfer")
col1_upload, col2_upload = st.columns(2)

content_image = col1_upload.file_uploader("Content image", type="jpg")
style_image = col2_upload.file_uploader("Style image", type="jpg")
col1_show, col2_show = st.columns(2)

with st.spinner('Wait upload content_image...'):
    if content_image is not None:
        image = Image.open(content_image)
        image = load_image(image)
        col1_show.image(image, caption='Content Image.', use_column_width=True)
    
    
with st.spinner('Wait upload style_image...'):    
    if style_image is not None:
        image2 = Image.open(style_image)
        image2 = load_image(image2)
        col2_show.image(image2, caption='Style image.', use_column_width=True)
    
    
if content_image is not None and style_image is not None:
    # st.write('Yea')
    with st.spinner('Wait for it...'):
        stylized_image = st.session_state.model(tf.constant(image), tf.constant(image2))[0]
        st.image(np.squeeze(stylized_image), caption='Result Image.', use_column_width=True)