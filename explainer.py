# %%
import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


#%%
def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = tf.keras.utils.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = tf.keras.utils.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array

def explainer(img, model):
    inp = cv2.resize(img, (224,224,))
    inp = inp.reshape((-1, 224, 224, 3))
    inp = tf.keras.applications.mobilenet_v2.preprocess_input(inp)

    # last_conv_layer  = model.get_layer('mobilenetv2_1.00_224').get_layer('Conv1')
    last_conv_layer  = model.get_layer('mobilenetv2_1.00_224').get_layer('out_relu')
    last_conv_layer_model = tf.keras.Model([model.get_layer('mobilenetv2_1.00_224').inputs], last_conv_layer.output)
    # last_conv_layer_model = tf.keras.Model(model.inputs, last_conv_layer.output)
    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in ["global_average_pooling2d", "dropout", "dense", 'dropout_1', 'dense_1']:
        x = model.get_layer(layer_name)(x)
    classifier_model = tf.keras.Model(classifier_input, x)

    with tf.GradientTape() as tape:
        # inputs = img[np.newaxis, ...]
        inp = cv2.resize(img, (224,224,))
        inp = inp.reshape((-1, 224, 224, 3))
        inputs = tf.keras.applications.mobilenet_v2.preprocess_input(inp)
        last_conv_layer_output = last_conv_layer_model(inputs)
        tape.watch(last_conv_layer_output)
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    grads = tape.gradient(top_class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # Average over all the filters to get a single 2D array
    gradcam = np.mean(last_conv_layer_output, axis=-1)
    # Clip the values (equivalent to applying ReLU)
    # and then normalise the values
    gradcam = np.clip(gradcam, 0, np.max(gradcam)) / np.max(gradcam)
    gradcam = cv2.resize(gradcam, (224, 224))
    # gradcam /= 100
    expanded_gradcam = np.expand_dims(gradcam, axis=-1)
    # subplot
    # fig, ax = plt.subplot(1,1,1)
    fig = plt.figure(figsize=(8,8))
    plt.rcParams["figure.autolayout"] = True
    # ax.imshow(cv2.resize(img, (224,224,))[:,:,0]*gradcam)
    plt.imshow((cv2.resize(img, (224,224,))*expanded_gradcam).astype(int), aspect='auto')
    plt.imshow(gradcam, alpha=0.2)
    plt.axis('off')

    return fig
    # return (expanded_gradcam*cv2.resize(img, (224,224,))).astype(int)
    # expanded_gradcam = np.broadcast_to(expanded_gradcam, cv2.resize(img, (224,224,)).shape)

    # plt.imshow(cv2.resize(img, (224,224,))[:,:,0]*gradcam)
    # plt.imshow(cv2.resize(img, (224,224,))[:,:,1]*gradcam)
    # plt.imshow(cv2.resize(img, (224,224,))*expanded_gradcam)
    plt.imshow((cv2.resize(img, (224,224,))*expanded_gradcam).astype(int))
    # plt.imshow(cv2.resize(img, (224,224,)))
    plt.imshow(gradcam, alpha=0.2)
    # plt.imshow(expanded_gradcam)
    plt.axis('off')
#%%
def make_gradcam_heatmap(img, model, last_conv_layer_name, pred_index=None):
    inp = cv2.resize(img, (224,224,))
    inp = inp.reshape((-1, 224, 224, 3))
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(inp)
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    last_conv_layer  = model.get_layer('mobilenetv2_1.00_224').get_layer('Conv1')
    grad_model = tf.keras.Model(
        model.get_layer('mobilenetv2_1.00_224').inputs, 
        [last_conv_layer.output,
          model.output])

    # grad_model = tf.keras.models.Model(
    #     [model.get_layer(last_conv_layer_name).inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    # )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


#%%
# heatmap = make_gradcam_heatmap(img, model, 'mobilenetv2_1.00_224')
# # heatmap = make_gradcam_heatmap(img, model, 'dense_1')
# #%%
# for layer in reversed(model.layers):
#   # check to see if the layer has a 4D output
#   if len(layer.output_shape) == 4:
#       print(layer.name)
#%%