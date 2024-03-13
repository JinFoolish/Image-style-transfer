import streamlit as st
from PIL import Image
# from trans import train_step, get_image
import os
import  numpy as np
import tensorflow as tf
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

    tf.random.set_seed(272)
    # pp = pprint.PrettyPrinter(indent=4)
    img_size = 400
    vgg = tf.keras.applications.VGG19(include_top=False,
                                    input_shape=(img_size, img_size, 3),
                                    weights='pretrained-model/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')

    vgg.trainable = False

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.03)  # tf26 version for adam optimizer

    STYLE_LAYERS = [
    ('block1_conv1', 0.5),
    ('block2_conv1', 0.8),
    ('block3_conv1', 1.7),
    ('block4_conv1', 1.5),
    ('block5_conv1', 2.0)]


    def compute_content_cost(content_output, generated_output):
        """
        Computes the content cost

        Arguments:
        a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C
        a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G

        Returns:
        J_content -- scalar that you compute using equation 1 above.
        """
        a_C = content_output[-1]
        a_G = generated_output[-1]

        # Retrieve dimensions from a_G (≈1 line)
        m, n_H, n_W, n_C = a_G.get_shape().as_list()

        # Reshape a_C and a_G (≈2 lines)
        a_C_unrolled = tf.reshape(a_C, shape=[m, n_H * n_W, n_C])  # Or tf.reshape(a_C, shape=[m, -1 , n_C])
        a_G_unrolled = tf.reshape(a_G, shape=[m, n_H * n_W, n_C])  # Or tf.reshape(a_G, shape=[m, -1 , n_C])

        # compute the cost with tensorflow (≈1 line)
        J_content = tf.reduce_sum(tf.square(a_C_unrolled - a_G_unrolled)) / (4.0 * n_H * n_W * n_C)
        return J_content


    def gram_matrix(A):
        """
        Argument:
        A -- matrix of shape (n_C, n_H*n_W)

        Returns:
        GA -- Gram matrix of A, of shape (n_C, n_C)
        """
        GA = tf.matmul(A, tf.transpose(A))
        return GA


    def compute_layer_style_cost(a_S, a_G):
        # Retrieve dimensions from a_G (≈1 line)
        m, n_H, n_W, n_C = a_G.get_shape().as_list()

        # Reshape the images to have them of shape (n_C, n_H*n_W) (≈2 lines)
        a_S = tf.transpose(tf.reshape(a_S, shape=[-1, n_C]))
        # OR a_S = tf.transpose(tf.reshape(a_S, shape=[ n_H * n_W, n_C]))
        a_G = tf.transpose(tf.reshape(a_G, shape=[-1, n_C]))
        # Computing gram_matrices for both images S and G (≈2 lines)
        GS = gram_matrix(a_S)
        GG = gram_matrix(a_G)

        # Computing the loss (≈1 line)
        J_style_layer = tf.reduce_sum(tf.square(GS - GG)) / (4.0 * ((n_H * n_W * n_C) ** 2))
        return J_style_layer

    def compute_style_cost(style_image_output, generated_image_output, STYLE_LAYERS=STYLE_LAYERS):
        """
        Computes the overall style cost from several chosen layers

        Arguments:
        style_image_output -- our tensorflow model
        generated_image_output --
        STYLE_LAYERS -- A python list containing:
                            - the names of the layers we would like to extract style from
                            - a coefficient for each of them

        Returns:
        J_style -- tensor representing a scalar value, style cost defined above by equation (2)
        """

        # initialize the overall style cost
        J_style = 0
        a_S = style_image_output[1:]

        a_G = generated_image_output[1:]
        for i, weight in zip(range(len(a_S)), STYLE_LAYERS):
            J_style_layer = compute_layer_style_cost(a_S[i], a_G[i])
            J_style += weight[1] * J_style_layer

        return J_style

    @tf.function()
    def total_cost(J_content, J_style, alpha=10, beta=40):
        J = alpha * J_content + beta * J_style
        return J

    def get_layer_outputs(vgg, layer_names):
        """ Creates a vgg model that returns a list of intermediate output values."""
        # layer_names has 'layer' elements in it.
        outputs = [vgg.get_layer(layer[0]).output for layer in layer_names]

        model = tf.keras.Model([vgg.input], outputs)
        return model

    def clip_0_1(image):
        return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

    # optimizer.build(variables)
    @tf.function()
    def train_step(vgg_model_outputs, a_S, a_C, generated_image, alpha=10, beta=40):
        with tf.GradientTape() as tape:
            a_G = vgg_model_outputs(generated_image)

            J_style = compute_style_cost(a_S, a_G)

            J_content = compute_content_cost(a_C, a_G)
            J = total_cost(J_content, J_style, alpha=alpha, beta=beta)

        grad = tape.gradient(J, generated_image)

        optimizer.apply_gradients([(grad, generated_image)])
        generated_image.assign(clip_0_1(generated_image))
        # For grading purposes
        return J

    def get_image(content_image, style_image,epochs=2500):
        content_image = np.array(content_image)
        content_image = tf.constant(np.reshape(content_image, ((1,) + content_image.shape)))

        style_image = np.array(style_image)
        style_image = tf.constant(np.reshape(style_image, ((1,) + style_image.shape)))

        generated_image = tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
        noise = tf.random.uniform(tf.shape(generated_image), 0, 0.8)
        generated_image = tf.add(generated_image, noise)
        generated_image = tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=1.0)

        content_layer = [('block5_conv4', 1)]

        vgg_model_outputs = get_layer_outputs(vgg, STYLE_LAYERS + content_layer)

        # content_target = vgg_model_outputs(content_image)  # Content encoder
        # style_targets = vgg_model_outputs(style_image)  # Style enconder

        preprocessed_content = tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
        a_C = vgg_model_outputs(preprocessed_content)

        # a_G = vgg_model_outputs(generated_image)

        # Compute the content cost
        # J_content = compute_content_cost(a_C, a_G)

        preprocessed_style = tf.Variable(tf.image.convert_image_dtype(style_image, tf.float32))
        a_S = vgg_model_outputs(preprocessed_style)

        # J_style = compute_style_cost(a_S, a_G)

        generated_image = tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
        return  vgg_model_outputs, a_S, a_C, generated_image

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
