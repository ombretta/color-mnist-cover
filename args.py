import argparse


def read_args():
    """ Reads command line arguments."""

    parser = argparse.ArgumentParser()
    # Add arguments
    parser.add_argument('--mode')
    parser.add_argument('--client')
    parser.add_argument('--host')
    parser.add_argument('--port')
    parser.add_argument('--n', type=int, default=10, help="Number of images to create.")
    parser.add_argument('--n_digits', type=int, default=150, help="Number of digits in one image.")
    parser.add_argument('--h', type=int, default=907, help="Image height.")
    parser.add_argument('--w', type=int, default=1321, help="Image width.")
    parser.add_argument('--digit_size', type=int, default=50, help="Size of each digit.")
    parser.add_argument('--background_color', type=str, default="black", help="Background color [black|white].")
    parser.add_argument('--n_hues', type=int, default=2, help="Number of samples hues.")
    parser.add_argument('--dataset_images', type=str, default="MNIST", help="Which images to use (e.g., MNIST, FashionMNIST).")
    parser.add_argument('--min_hue_distance', type=int, default=30, help="Minimum hue distance for different sampled hues.")
    parser.add_argument('--max_hue_noise', type=int, default=3, help="Maximum hue variation for the same sampled hue.")
    parser.add_argument('--saturation_range', nargs='+', type=float, default=[0.5, 1.0], help="Range for saturation variations.")
    parser.add_argument('--max_saturation_noise', type=float, default=0.05, help="Maximum saturation noise.")
    parser.add_argument('--value_range', nargs='+', type=float, default=[0.5, 1.0], help="Range for value variations.")
    parser.add_argument('--max_value_noise', type=float, default=0.05, help="Maximum value noise.")
    parser.add_argument('--root', type=str, default="datasets/colorMNIST/", help="Root path for the dataset folder.")
    parser.add_argument('--datadir', type=str, default="images/", help="Where to save the created images.")
    args = parser.parse_args()
    return args

