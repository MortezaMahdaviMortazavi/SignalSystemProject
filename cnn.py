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

def plot_outputs(outputs):
    num_outputs = len(outputs)
    fig, axes = plt.subplots(1, num_outputs, figsize=(12, 3))  # Adjust the figsize as per your preference

    for i in range(num_outputs):
        output = outputs[i]
        output_channels = output.shape[2]
        
        
        if output_channels == 1:
            axes[i].imshow(output[:, :, 0], cmap='gray')
        else:
            try:
                axes[i].imshow(output)
            except:
                # raise Exception(f"Number of channels are not valid and its {output_channels}")
                pass

        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    bengio_path = 'images/bengio.jpg'
    goodfellow_path = 'images/goodfellow.png'
    hinton_path = 'images/hinton.jpg'
    lecun_path = 'images/lecun.jpg'

    bengio_img = np.array(Image.open(bengio_path))
    goodfellow_img = np.array(Image.open(goodfellow_path))
    hinton_img = np.array(Image.open(hinton_path))
    lecun_img = np.array(Image.open(lecun_path))

    """Layer 1"""
    bottem_sobel = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]]
    )

    top_sobel = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])

    sharpening = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    weighted_averaging_3x3 = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ])

    """Layer 2"""
    robert_x = np.array([
        [1, 0],
        [0, -1]
    ])

    robert_y = np.array([
        [0, 1],
        [-1, 0]
    ])

    averaging_2x2 = np.array([
        [1, 1],
        [1, 1]
    ])

    """Layer 3"""
    gaussian_5x5 = np.array([
        [1, 4, 7, 4, 1],
        [4, 16, 26, 16, 4],
        [7, 26, 41, 26, 7],
        [4, 16, 26, 16, 4],
        [1, 4, 7, 4, 1]
    ])

    averaging_5x5 = np.ones_like(gaussian_5x5) / 25

    # Example usage with a grayscale image
    l1_filters = []
    l1_filters.append(bottem_sobel.reshape(3,3,1))
    l1_filters.append(top_sobel.reshape(3,3,1))
    l1_filters.append(sharpening.reshape(3,3,1))

    l1_weighted_average = []
    l1_weighted_average.append(weighted_averaging_3x3.reshape(3,3,1))

    l2_filters = []
    l2_filters.append(robert_x.reshape(2,2,1))
    l2_filters.append(robert_y.reshape(2,2,1))
    l2_filters.append(averaging_2x2.reshape(2,2,1))

    l3_filters = []
    l3_filters.append(gaussian_5x5.reshape(5,5,1))
    l3_filters.append(averaging_5x5.reshape(5,5,1))

    layers = [
        {'filters':l1_filters,'stride':1,'padding':'same'},
        {'filters':l1_weighted_average,"stride":1,'padding':'same'},
        {'filters':l2_filters,"stride":2,'padding':'valid'},
        {'filters':l3_filters,"stride":1,'padding':'same'},
    ]


    outputs = apply_conv_layers(bengio_img,layers)
    plot_outputs(outputs=outputs)


