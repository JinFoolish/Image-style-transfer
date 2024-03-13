import sys
# import matplotlib.pyplot as plt
# from matplotlib.pyplot import imshow  # imshow用于显示二维图像数据
from PIL import Image  # Python Imaging Library,用于图像处理的库，提供了广泛的图像操作功能
import numpy as np
import tensorflow as tf
# import pprint

tf.random.set_seed(272)
# pp = pprint.PrettyPrinter(indent=4)
img_size = 400
vgg = tf.keras.applications.VGG19(include_top=False,
                                  input_shape=(img_size, img_size, 3),
                                  weights='pretrained-model/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')

vgg.trainable = False
# pp.pprint(vgg)

TEST = False
if TEST:
    args = ['style/la_muse.jpg', 'style/my_style1.png', 'data/output.jpg']
else:
    args = sys.argv[1:]

content_image_path = "data/content.jpg"
style_image_path = "data/style.jpg"
save_file_path = "data/save.jpg"
# content_image = Image.open("images/louvre.jpg").convert('RGB')
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
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G

    Returns:
    J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
    """

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


# print(vgg.get_layer('block5_conv4').output)


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

    # Set a_S to be the hidden layer activation from the layer we have selected.
    # The first element of the array contains the input layer image, which must not to be used.
    a_S = style_image_output[1:]

    # Set a_G to be the output of the choosen hidden layers.
    # The First element of the list contains the input layer image which must not to be used.
    a_G = generated_image_output[1:]
    for i, weight in zip(range(len(a_S)), STYLE_LAYERS):
        # Compute style_cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S[i], a_G[i])

        # Add weight * J_style_layer of this layer to overall style cost
        J_style += weight[1] * J_style_layer

    return J_style


@tf.function()
def total_cost(J_content, J_style, alpha=10, beta=40):
    """
    Computes the total cost function

    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost

    Returns:
    J -- total cost as defined by the formula above.
    """
    J = alpha * J_content + beta * J_style
    return J


# print(generated_image.shape)
# imshow(generated_image.numpy()[0])
# plt.show()
# plt.pause(2)
# plt.close()

def get_layer_outputs(vgg, layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    # layer_names has 'layer' elements in it.
    outputs = [vgg.get_layer(layer[0]).output for layer in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model



def clip_0_1(image):
    """
    Truncate all the pixels in the tensor to be between 0 and 1

    Arguments:
    image -- Tensor
    J_style -- style cost coded above

    Returns:
    Tensor
    """
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


def tensor_to_image(tensor):
    """
    Converts the given tensor into a PIL image

    Arguments:
    tensor -- Tensor

    Returns:
    Image: A PIL image
    """
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)


# UNQ_C5
# GRADED FUNCTION: train_step
# optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.03)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.03)  # tf26 version for adam optimizer


# optimizer.build(variables)
@tf.function()
def train_step(vgg_model_outputs, a_S, a_C, generated_image, alpha=10, beta=40):
    with tf.GradientTape() as tape:
        # In this function you must use the precomputed encoded images a_S and a_C
        # Compute a_G as the vgg_model_outputs for the current generated image

        ### START CODE HERE

        # (1 line)
        a_G = vgg_model_outputs(generated_image)

        # Compute the style cost
        # (1 line)
        J_style = compute_style_cost(a_S, a_G)

        # (2 lines)
        # Compute the content cost
        J_content = compute_content_cost(a_C, a_G)
        # Compute the total cost
        J = total_cost(J_content, J_style, alpha=alpha, beta=beta)

        ### END CODE HERE

    grad = tape.gradient(J, generated_image)

    optimizer.apply_gradients([(grad, generated_image)])
    generated_image.assign(clip_0_1(generated_image))
    # For grading purposes
    return J

def get_image(content_image, style_image,epochs=2500):
    content_image = np.array(content_image)
    # content_image = np.array(Image.open(content_image_path).convert('RGB').resize((img_size, img_size)))
    content_image = tf.constant(np.reshape(content_image, ((1,) + content_image.shape)))

    style_image = np.array(style_image)
    # style_image = np.array(Image.open(style_image_path).resize((img_size, img_size)))
    style_image = tf.constant(np.reshape(style_image, ((1,) + style_image.shape)))

    generated_image = tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
    noise = tf.random.uniform(tf.shape(generated_image), 0, 0.8)
    generated_image = tf.add(generated_image, noise)
    generated_image = tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=1.0)

    content_layer = [('block5_conv4', 1)]

    vgg_model_outputs = get_layer_outputs(vgg, STYLE_LAYERS + content_layer)

    content_target = vgg_model_outputs(content_image)  # Content encoder
    style_targets = vgg_model_outputs(style_image)  # Style enconder

    # Assign the content image to be the input of the VGG model.
    # Set a_C to be the hidden layer activation from the layer we have selected
    preprocessed_content = tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
    a_C = vgg_model_outputs(preprocessed_content)

    # Set a_G to be the hidden layer activation from same layer. Here, a_G references model['conv4_2']
    # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input.
    a_G = vgg_model_outputs(generated_image)

    # Compute the content cost
    J_content = compute_content_cost(a_C, a_G)
    # print(J_content)

    # Assign the input of the model to be the "style" image
    preprocessed_style = tf.Variable(tf.image.convert_image_dtype(style_image, tf.float32))
    a_S = vgg_model_outputs(preprocessed_style)

    # Compute the style cost
    J_style = compute_style_cost(a_S, a_G)
    # print(J_style)
    generated_image = tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
    return  vgg_model_outputs, a_S, a_C, generated_image

def run(content_image, style_image,epochs=2500):
    vgg_model_outputs, a_S, a_C, generated_image = get_image(content_image,style_image)
    for i in range(epochs):
        train_step(vgg_model_outputs, a_S, a_C,generated_image, alpha=100, beta=10 ** 2)
    return generated_image[0]

# fig = plt.figure()
# content_image = np.array(Image.open(content_image_path).convert('RGB').resize((img_size, img_size)))
# style_image = np.array(Image.open(style_image_path).resize((img_size, img_size)))
# imshow(run(content_image, style_image))
# plt.axis('off')
# plt.savefig(save_file_path, bbox_inches='tight', pad_inches=0)
# plt.close(fig)