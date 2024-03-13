import streamlit as st
from PIL import Image
from trans import train_step, get_image
import os
import  numpy as np
img_size = 400
st.set_page_config(layout="wide", page_title="Image Transfer")

st.write("## 图片风格转换")
st.write(
    ":dog: Try uploading an image to watch the background magically removed. Full quality images can be downloaded from the sidebar. This code is open source and available [here](https://github.com/tyler-simons/BackgroundRemoval) on GitHub. Special thanks to the [rembg library](https://github.com/danielgatis/rembg) :grin:"
)
bar = st.progress(0.0, text="")
st.sidebar.write("## Upload and download :gear:")

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# Download the fixed image
def convert_image():
    if content_upload is None:
        content_image = Image.open("./data/content.jpg").convert('RGB').resize((img_size, img_size))
    else:
        content_image = Image.open(content_upload).convert('RGB').resize((img_size, img_size))
    if st.session_state['style_path'] != '':
        style_image = Image.open(st.session_state['style_path']).resize((img_size, img_size))
    else:
        style_image = Image.open("./data/style.jpg").resize((img_size, img_size))

    vgg_model_outputs, a_S, a_C, generated_image = get_image(content_image, style_image)
    bar_interval = steps//10
    for i in range(steps):
        train_step(vgg_model_outputs, a_S, a_C, generated_image, alpha=100, beta=10 ** 2)
        if i%bar_interval==0:
            bar.progress(i/steps, text="处理ing")
    bar.progress(1.0)
    image = generated_image[0].numpy()
    col3.write("艺术化图像")
    col3.image(image)

def show_image(col, upload, title):
    image = Image.open(upload)
    col.write(title)
    col.image(image)


col1, col2, col3 = st.columns(3)

content_upload = st.sidebar.file_uploader("上传内容图像", type=["png", "jpg", "jpeg"])
if content_upload is not None:
    if content_upload.size > MAX_FILE_SIZE:
        st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
    else:
        show_image(col1, content_upload, "内容图像")
else:
    show_image(col1, "./data/content.jpg", "内容图像")

st.session_state['style_path'] = "style/la_muse.jpg"
st.session_state['pre_style'] = "无"
st.session_state['style_upload'] = None

def set_style_path():
    p = '无'
    if pre_style!=st.session_state['pre_style']:
        p = os.path.join("style",pre_style)
    st.session_state['pre_style'] = pre_style

    if style_upload!=st.session_state['style_upload']:
        p = style_upload
    if p!='无':
        st.session_state['style_path'] = p
    print(st.session_state['pre_style'])


pre_style = st.sidebar.selectbox("选择提供的风格图像",options=["无"]+[i for i in os.listdir("style")])


style_upload = st.sidebar.file_uploader("上传风格图像", type=["png", "jpg", "jpeg"])
# show_image(col2, "./data/style.jpg", "风格图像")
# set_style_path()
p = '无'
if pre_style!=st.session_state['pre_style']:
    p = os.path.join("style",pre_style)
st.session_state['pre_style'] = pre_style

if style_upload!=st.session_state['style_upload']:
    p = style_upload
if p!='无':
    st.session_state['style_path'] = p
print(st.session_state['pre_style'])
show_image(col2, st.session_state['style_path'], "风格图像")
steps = st.sidebar.slider("处理次数", min_value=10, max_value=4000, step=1)

run_button = st.sidebar.button("艺术化处理",
                               on_click=convert_image,
                                    type="primary")
