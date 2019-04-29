import numpy as np
import os
from PIL import Image

import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.python.keras.preprocessing import image as kp_image
from tensorflow.python.keras import models

import time


# convolutional layers in our pre-trained network relevant to content
content_layers = ['block5_conv2']

# convolutional layers in our pre-trained network relevant to style
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

image_max_dim = 512
num_content_layers = len(content_layers)
num_style_layers = len(style_layers)


def get_processed_image_tensor(path, max_dim):
    """Open an image at path and return a pre-processed tensor scaled to max_dim"""
    image = Image.open(path)
    image_dim = max(image.size)
    scale_factor = max_dim / image_dim

    image = image.resize((round(image.size[0] * scale_factor), round(image.size[1] * scale_factor)))
    image = kp_image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.vgg19.preprocess_input(image)

    return image


def get_unprocessed_image(processed_image):
    """Undo pre-processing in order to display images generated from our network"""
    image = processed_image.copy()
    if len(image.shape) == 4:
        image = np.squeeze(image, 0)

    assert len(image.shape) == 3, 'Input must be of dimension [1, height, width, channel]' \
                                  ' or [height, width, channel]'

    # Undo VGG image pre-processing
    image[:, :, 0] += 103.939
    image[:, :, 1] += 116.779
    image[:, :, 2] += 123.68
    image = image[:, :, ::-1]

    # keep pixel values between 0 and 255
    image = np.clip(image, 0, 255).astype('uint8')

    return image


def create_model():
    """Return a VGG19 network pre-trained on the imagenet database"""
    vgg19 = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
    vgg19.trainable = False

    style_outputs = [vgg19.get_layer(name).output for name in style_layers]
    content_outputs = [vgg19.get_layer(name).output for name in content_layers]
    outputs = style_outputs + content_outputs

    return models.Model(vgg19.input, outputs)


def get_content_loss(input_content, target):
    return tf.reduce_mean(tf.square(input_content - target))


def get_gram_matrix(tensor):
    """Return the Gram matrix of a tensor"""
    channels = int(tensor.shape[-1])
    a = tf.reshape(tensor, [-1, channels])
    n = tf.shape(a)[0]

    gram = tf.matmul(a, a, transpose_a=True)
    gram /= tf.cast(n, tf.float32)

    return gram


def get_style_loss(input_style, target_gram):
    input_gram = get_gram_matrix(input_style)

    return tf.reduce_mean(tf.square(input_gram - target_gram))


def get_feature_representations(model, content_path, style_path):
    """Function to pre-process our content and style images and return their feature layers"""
    content_image = get_processed_image_tensor(content_path, image_max_dim)
    style_image = get_processed_image_tensor(style_path, image_max_dim)

    content_outputs = model(content_image)
    style_outputs = model(style_image)

    style_features = [style_layer[0] for style_layer in style_outputs[:num_style_layers]]
    content_features = [content_layer[0] for content_layer in content_outputs[num_style_layers:]]

    return style_features, content_features


def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
    """This function will compute the loss

      Arguments:
        model: The model that will give us access to the intermediate layers
        loss_weights: The weights of contribution between style and content
        init_image: The image we are updating with our optimization process
        gram_style_features: Gram matrices corresponding to the defined style layers
        content_features: Outputs from defined content layers

      Returns:
        returns the total loss, style loss, and content loss
    """
    style_weight, content_weight = loss_weights
    model_outputs = model(init_image)

    style_output_features = model_outputs[:num_style_layers]
    content_output_features = model_outputs[num_style_layers:]

    style_score = 0
    content_score = 0

    weight_per_style_layer = 1.0 / float(num_style_layers)

    for target_style, comb_style in zip(gram_style_features, style_output_features):
        style_score += weight_per_style_layer * get_style_loss(comb_style[0], target_style)

    weight_per_content_layer = 1.0 / float(num_content_layers)

    for target_content, comb_content in zip(content_features, content_output_features):
        content_score += weight_per_content_layer * get_content_loss(comb_content[0], target_content)

    style_score *= style_weight
    content_score *= content_weight

    loss = style_score + content_score

    return loss, style_score, content_score


def compute_grads(cfg):
    with tf.GradientTape() as gt:
        loss = compute_loss(**cfg)

    total_loss = loss[0]

    return gt.gradient(total_loss, cfg['init_image']), loss


def run_style_transfer(content_path,
                       style_path,
                       num_iterations=1000,
                       content_weight=1e3,
                       style_weight=1e-2):
    """This function will compute the loss

      Arguments:
        content_path: Path to the content image
        style_path: Path to the style image
        num_iterations: Number of iterations to perform
        content_weight: The scale of loss derived from the content image
        style_weight: The scale of loss derived from the style image

      Returns:
        returns the best image and its total loss
    """

    # Make sure we don't train any of our layers
    model = create_model()
    for layer in model.layers:
        layer.trainable = False

    # Get the style and content feature representations as well as the gram matrices for the style features
    style_features, content_features = get_feature_representations(model, content_path, style_path)
    gram_style_features = [get_gram_matrix(style_feature) for style_feature in style_features]

    # Set initial image
    init_image = get_processed_image_tensor(content_path, image_max_dim)
    init_image = tfe.Variable(init_image, dtype=tf.float32)

    # Create our optimizer
    opt = tf.train.AdamOptimizer(learning_rate=5, beta1=0.99, epsilon=1e-1)

    # Set our initial variables
    best_loss, best_img = float('inf'), None

    # Create a nice config
    loss_weights = (style_weight, content_weight)
    cfg = {
        'model': model,
        'loss_weights': loss_weights,
        'init_image': init_image,
        'gram_style_features': gram_style_features,
        'content_features': content_features
    }

    # Determine an interval to display data
    num_rows = 2
    num_cols = 5
    display_interval = num_iterations / (num_rows * num_cols)
    global_start = time.time()

    norm_means = np.array([103.939, 116.779, 123.68])
    min_values = -norm_means
    max_values = 255 - norm_means

    images = []
    for i in range(num_iterations):
        grads, all_loss = compute_grads(cfg)
        loss, style_score, content_score = all_loss
        opt.apply_gradients([(grads, init_image)])
        clipped = tf.clip_by_value(init_image, min_values, max_values)
        init_image.assign(clipped)

        if loss < best_loss:
            # Update best loss and best image from total loss.
            best_loss = loss
            best_img = get_unprocessed_image(init_image.numpy())

        if i % display_interval == 0:
            start_time = time.time()

            plot_img = init_image.numpy()
            plot_img = get_unprocessed_image(plot_img)
            images.append(plot_img)
            print('Iteration: {}'.format(i))
            print('Total loss: {:.4e}, '
                  'style loss: {:.4e}, '
                  'content loss: {:.4e}, '
                  'time: {:.4f}s'.format(loss, style_score, content_score, time.time() - start_time))
    print('Total time: {:.4f}s'.format(time.time() - global_start))

    return best_img, best_loss


cur_content_path = 'images/waterways.jpg'
cur_style_path = 'images/starry_night.jpg'
out_image_path = 'new_images/waterways_starry.jpg'


if __name__ == "__main__":

    # Make sure we can find our video card
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    tf.enable_eager_execution()

    best, best_loss = run_style_transfer(cur_content_path,
                                         cur_style_path)

    img = Image.fromarray(best)
    img.save(out_image_path)


