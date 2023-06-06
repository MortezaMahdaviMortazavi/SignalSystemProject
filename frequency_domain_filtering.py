import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

from PIL import Image
from tqdm import tqdm
from cnn import apply_conv_layers

def fourier_transform(inp):
    fourier = np.fft.fft2(inp)
    fourier_shifted = np.fft.fftshift(fourier)
    magnitude_spectrum = np.abs(fourier_shifted)
    # phase = np.angle(fourier_shifted)
    phase_spectrum = np.angle(fourier_shifted)
    return magnitude_spectrum,phase_spectrum

def apply_fourier_transform(outputs):
    filters_ft_output = []
    for output in outputs:
        out_magn,out_phase = fourier_transform(outputs)
        filters_ft_output.append(
            {
                'original':output,
                'magnitude':out_magn,
                'phase':out_phase
            }
        )

    return filters_ft_output


def plot_magnitude_and_phase(outputs):
    num_items = len(outputs)

    # Create subplots
    fig, axes = plt.subplots(num_items, 2, figsize=(10, 5*num_items))
    fig.tight_layout()

    # Iterate over each item in the outputs list
    for i, output in enumerate(outputs):
        # Get the magnitude and phase spectra for the current item
        magnitude_spectrum = output['magnitude']
        phase_spectrum = output['phase']

        # Plot the magnitude spectrum
        axes[i, 0].imshow(np.log(1 + magnitude_spectrum), cmap='gray')
        axes[i, 0].set_title(f'Magnitude - Item {i+1}')
        axes[i, 0].axis('off')

        # Plot the phase spectrum
        axes[i, 1].imshow(phase_spectrum, cmap='gray')
        axes[i, 1].set_title(f'Phase - Item {i+1}')
        axes[i, 1].axis('off')

    plt.show()



if __name__ == '__main__':
    # Load the image
    bengio = np.array(cv.imread('images/bengio.jpg'))
    goodfellaw = np.array(cv.imread('images/goodfellaw.png'))
    hinton = np.array(cv.imread('images/hinton.jpg'))
    lecun = np.array(cv.imread('images/lecun.jpg'))


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
    outputs = apply_conv_layers(bengio,layers)
    apply_fourier_transform_outputs = apply_fourier_transform(outputs)
    plot_magnitude_and_phase(apply_fourier_transform_outputs)
