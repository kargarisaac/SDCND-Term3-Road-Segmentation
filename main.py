
# coding: utf-8

import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

get_ipython().run_line_magic('matplotlib', 'inline')

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

REG_ = 1e-5
STD_ = 1e-2

def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    graph = tf.get_default_graph()

    input_tensor = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out_tensor = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out_tensor = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out_tensor = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return input_tensor, keep_prob, layer3_out_tensor, layer4_out_tensor, layer7_out_tensor
tests.test_load_vgg(load_vgg, tf)

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output

    Hint:
    you need 1x1 convolution for the skip connections to transform # of filters.
    On the output of the 3rd layer you have 256 filters, 4th - 512 layers; layers
    of decoder has 2 filters (as the amount of classes in the segmentation).
    1x1 convolutions makes a transformation from 256 (512) filters to 2.

    """
    # TODO: Implement function
    layer7_out_conv_1x1 = tf.layers.conv2d(vgg_layer7_out,
                                           num_classes,
                                           1,
                                           padding='same',
                                           kernel_initializer = tf.truncated_normal_initializer(stddev=STD_),
                                           kernel_regularizer = tf.contrib.layers.l2_regularizer(REG_))

    upsampling1 = tf.layers.conv2d_transpose(layer7_out_conv_1x1,
                                            num_classes,
                                            4,
                                            2,
                                            padding='same',
                                            kernel_initializer = tf.truncated_normal_initializer(stddev=STD_),
                                            kernel_regularizer = tf.contrib.layers.l2_regularizer(REG_))

#     vgg_layer4_out_scaled = tf.multiply(vgg_layer4_out, 0.01)
    vgg_layer4_out_conv_1x1 = tf.layers.conv2d(vgg_layer4_out,
                                               num_classes,
                                               1,
                                               padding= 'same',
                                               kernel_initializer = tf.truncated_normal_initializer(stddev=STD_),
                                               kernel_regularizer = tf.contrib.layers.l2_regularizer(REG_))


    skip1 = tf.add(upsampling1, vgg_layer4_out_conv_1x1)

    upsampling2 = tf.layers.conv2d_transpose(skip1,
                                            num_classes,
                                            4,
                                            2,
                                            padding='same',
                                            kernel_initializer = tf.truncated_normal_initializer(stddev=STD_),
                                            kernel_regularizer = tf.contrib.layers.l2_regularizer(REG_))

#     vgg_layer3_out_scaled = tf.multiply(vgg_layer3_out, 0.0001)
    vgg_layer3_out_conv_1x1 = tf.layers.conv2d(vgg_layer3_out,
                                               num_classes,
                                               1,
                                               padding= 'same',
                                               kernel_initializer = tf.truncated_normal_initializer(stddev=STD_),
                                               kernel_regularizer = tf.contrib.layers.l2_regularizer(REG_))


    skip2 = tf.add(upsampling2, vgg_layer3_out_conv_1x1)

    upsampling3 = tf.layers.conv2d_transpose(skip2,
                                            num_classes,
                                            16, 8,
                                            padding='same',
                                            kernel_initializer = tf.truncated_normal_initializer(stddev=STD_),
                                            kernel_regularizer = tf.contrib.layers.l2_regularizer(REG_))

    return upsampling3
tests.test_layers(layers)

def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    logits = tf.reshape(nn_last_layer, (-1, num_classes))

    correct_label = tf.reshape(correct_label, (-1, num_classes))

    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))

    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    cross_entropy_loss = cross_entropy_loss + sum(reg_losses)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    train_op = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    sess.run(tf.global_variables_initializer())

    print('Training started...')

    for ep in range(epochs):

        print('%d / %d : '%(ep,epochs))

        for image, label in get_batches_fn(batch_size):

            _, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict={input_image:image, correct_label:label, keep_prob:0.5, learning_rate:1e-4})

            print('loss: %.4f '%(loss))

        print("\n")

tests.test_train_nn(train_nn)

def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
#     helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    ## placeholders
    correct_label = tf.placeholder(tf.float32, shape=(None, None, None, num_classes), name='correct_label')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    epochs = 50
    batch_size = 8

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        layer_output = layers(layer3_out, layer4_out, layer7_out, num_classes)
        logits, train_op, cross_entropy_loss = optimize(layer_output, correct_label, learning_rate, num_classes)

        # TODO: Train NN using the train_nn function

        train_nn(sess, epochs, batch_size, get_batches_fn, train_op,
                 cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate)

        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        saver = tf.train.Saver()
        saver.save(sess, 'model')
        # OPTIONAL: Apply the trained model to a video
#         run follwing command in terminal in the images directory
#         ffmpeg -framerate 5 -i um_%06d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p output.mp4


if __name__ == '__main__':
    run()


# ### run follwing command in terminal in the images directory:
#
# ffmpeg -framerate 5 -i um_%06d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p output.mp4
#

# ## Hint
#
# https://askubuntu.com/questions/610903/how-can-i-create-a-video-file-from-a-set-of-jpg-images
#
# This is the command all together:
#
# ffmpeg -framerate 25 -i image-%05d.jpg -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p output.mp4
#
# Let me break it down:
#
# -framerate
#
# is the number of frames (images) per second
#
# -i scene_%05d.jpg
#
# this determines the file name sequence it looks for. image- means all the files start with this. 0 is the number repeated, and the 5 is number of times (so it is looking for any file starting at image-00000.jpg. The d is telling it to count up in whole numbers, so the files it will detect are everything from image-00001 to image-99999.
#
# -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p
#
# -c:v libx264 - the video codec is libx264 (H.264).
#
# -profile:v high - use H.264 High Profile (advanced features, better quality).
#
# -crf 20 - constant quality mode, very high quality (lower numbers are higher quality, 18 is the smallest you would want to use).
#
# -pix_fmt yuv420p - use YUV pixel format and 4:2:0 Chroma subsampling
#
# output.mp4
#
# The file name (output.mp4)
#
# Remember that ffmpeg needs a continuous sequence of images to load in. If it jumps from image-00001 to image-00003 it will stop.
#
# If you images are like this:
#
# image-1
# image-2
# ...
# image-35
# then change the -i part to -i image-%00d.
#
# Update. Your edit says the pattern is image-01.jpg to image-02.jpg. That means you need the  image-%02d.jpg pattern.
