import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def MaxPooling2D(input_array, window, stride):
    input_height, input_width, input_channels = input_array.shape

    window_height, window_width = window
    # stride_height, stride_width = stride

    output_height = (input_height - window_height) // stride + 1
    output_width = (input_width - window_width) // stride + 1

    output_array = np.zeros((output_height, output_width, input_channels))

    for i in range(input_channels):
        for j in range(output_height):
            for k in range(output_width):
                slice_start_h = j * stride
                slice_end_h = slice_start_h + window_height
                slice_start_w = k * stride
                slice_end_w = slice_start_w + window_width

                output_array[j, k, i] = np.max(input_array[slice_start_h:slice_end_h, slice_start_w:slice_end_w, i])

    return output_array


def Conv2D(img, filters, stride, padding):
    assert padding in ['same', 'valid']

    img_height, img_width, img_channels = img.shape
    filters_height, filters_width, filters_channels = filters[0].shape
    number_of_filters = len(filters)

    if padding == 'valid':
        pass  # Nothing changes

    elif padding == 'same':
        pad_height = max((img_height - 1) * stride + filters_height - img_height, 0)
        pad_width = max((img_width - 1) * stride + filters_width - img_width, 0)
        img = np.pad(img, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')
        img_height, img_width, img_channels = img.shape

    out_height = (img_height - filters_height) // stride + 1
    out_width = (img_width - filters_width) // stride + 1

    output_array = np.zeros((out_height, out_width, number_of_filters))

    for i in range(number_of_filters):
        filter_i = filters[i]

        for j in range(out_height):
            for k in range(out_width):
                slice_start_h = j * stride
                slice_end_h = slice_start_h + filters_height
                slice_start_w = k * stride
                slice_end_w = slice_start_w + filters_width

                output_array[j, k, i] = np.sum(img[slice_start_h:slice_end_h, slice_start_w:slice_end_w, :] * filter_i)

    return output_array

def apply_conv_layers(img,
                      conv_layers,
                      window=(2,2),
                      pooling_stride=2):
    output = img
    outputs = []

    for layer in conv_layers:
        filters = layer['filters']
        stride = layer['stride']
        padding = layer['padding']
        output = Conv2D(output, filters, stride, padding)
        outputs.append(output)

    output = MaxPooling2D(output,window=window,stride=pooling_stride)
    outputs.append(output)

    return outputs
