import json

import matplotlib.pyplot as plt
import numpy as np
import requests
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
from streamlit_drawable_canvas import st_canvas
from sklearn import datasets


@st.cache
def load_data():
    data = datasets.load_digits()
    return data["images"], data["target"]


images, targets = load_data()

st.header("Using original dataset")

col1, col2 = st.columns(2)



with col1:
    st.subheader("Input")
    sample_select = st.slider("sample", min_value=0, max_value=len(targets))
    st.write(f"sample index: {sample_select}   target value: {targets[sample_select]}")
    st.image((255 - images[sample_select] / 16 * 255).round().astype("uint8"), width=200)

data = images[sample_select].flatten().tolist()

with col2:
    st.subheader("Model Predict")
    res = requests.post("http://api/predict", json={"data": [data]})
    st.write(res.json())

st.header("Using canvas")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Input")
    st.write("draw a digit from 0-9 on canvas, try to fill all space")

    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=17,
        stroke_color="#000000",
        background_color="#ffffff",
        background_image=None,
        update_streamlit=True,
        height=200,
        width=200,
        drawing_mode="freedraw",
        key="canvas",
    )

    img_result = st.empty()

image_data = canvas_result.image_data[:, :, 0]

# Cropping image
white = np.argwhere(image_data < 255)
crop = 0
if len(white) > 0:
    height, width = image_data.shape
    top, bottom = white[0, 0], white[-1, 0]
    left_edge = white[:, 1].min()
    right_edge =  white[:, 1].max()
    draw_w = right_edge - left_edge
    x = (bottom - top - draw_w) // 2
    col = np.repeat(255, height).reshape(-1, 1)
    if x > left_edge:
        diff = x - left_edge
        image_data = np.c_[np.tile(col, diff), image_data]
    else:
        diff = left_edge - x
        image_data = image_data[:, diff:]
    if x + draw_w < right_edge:
        diff = right_edge - x - draw_w 
        image_data = np.c_[image_data, np.tile(col, diff)]
    else:
        diff = width - draw_w - x + right_edge
        image_data = image_data[:, :diff]

im = Image.fromarray(image_data.astype("uint8"), mode="L")

RESAMPLE = {
    "BICUBIC": Image.BICUBIC,
    "NEAREST": Image.NEAREST,
    "BOX": Image.BOX,
    "BILINEAR": Image.BILINEAR,
    "HAMMING": Image.HAMMING,
    "LANCZOS": Image.LANCZOS,
}

FILTERS = {
    "BLUR": ImageFilter.BLUR,
    "SMOOTH": ImageFilter.SMOOTH,
    "CONTOUR": ImageFilter.CONTOUR,
    "DETAIL": ImageFilter.DETAIL,
    "EDGE_ENHANCE": ImageFilter.EDGE_ENHANCE,
    "EDGE_ENHANCE_MORE": ImageFilter.EDGE_ENHANCE_MORE,
    "EMBOSS": ImageFilter.EMBOSS,
    "FIND_EDGES": ImageFilter.FIND_EDGES,
    "SHARPEN": ImageFilter.SHARPEN,
    "SMOOTH_MORE": ImageFilter.SMOOTH_MORE,
}

filter_ = st.sidebar.selectbox("filter", FILTERS)
im_filter = im.filter(FILTERS[filter_])

resample = st.sidebar.selectbox("resample", RESAMPLE)
im_resized = im_filter.resize((8, 8), resample=RESAMPLE[resample])


contrast = st.sidebar.slider("contrast", min_value=1.0, max_value=5.0, step=1.0, value=2.0)
sharpness = st.sidebar.slider("shapness", min_value=1.0, max_value=5.0, step=1.0, value=1.0)
im_enhance = ImageEnhance.Sharpness(im_resized).enhance(sharpness)
im_enhance = ImageEnhance.Contrast(im_enhance).enhance(contrast)

with img_result:
    st.write("resized image 8x8")
    st.image(im_enhance, width=200)


im_array = np.array(im_enhance)
im_array = ((255 - im_array) / 255 * 16).round().astype("uint8")
data = im_array.flatten().tolist()

with col2:
    st.subheader("Model Predict")
    res = requests.post("http://api/predict", json={"data": [data]})
    st.write(res.json())
