import os
import yaml

import numpy as np
import torch

import torchvision
from torchvision.utils import save_image

import colorsys

from args import read_args

"""
This script generates colorful images of MNIST digits with varying amounts of hue variations: 
1) One hue (with varying brightness and saturation)
2) Two hues (with varying brightness and saturation)
3) Three hues (with varying brightness and saturation)
4) Four hues (with varying brightness and saturation)
5) Five hues (with varying brightness and saturation)
...
"""


def write_configs(args):
    """
    :param datadir: the directory where the images are saved
    :param n: number of images
    :param h: image height
    :param w: image width
    :param n_hues: number of hues in one image
    :return:
    """

    # Save arguments to YAML file
    with open(args.datadir + "data_properties.txt", 'w') as file:
        yaml.dump(vars(args), file)


def create_data_dir(args):
    """Create the dataset folder."""

    print("args", args)
    datadir = args.datadir

    if not os.path.exists(datadir):
        os.mkdir(datadir)
        write_configs(args)

    return datadir



def sample_colors(n_digits=1, n_hues=1, min_hue_distance=40, saturation_range=[0.5, 1], value_range=[0.5, 1],
                  max_hue_noise=3, max_saturation_noise=0.05, max_value_noise=0.05, **args):
    """This function samples colors to assign to the digits.
       :param n_digits: the number of digits appearing in one image
       :param n_hues: the number of different hues to include."""

    # Sample hue, saturation and value
    chosen_hues = torch.ones(n_hues)
    saturation = torch.rand(n_hues) * (saturation_range[1]-saturation_range[0]) + saturation_range[0]  # Guarantees minimum saturation
    value = torch.rand(n_hues) * (value_range[1]-value_range[0]) + value_range[0]  # Guarantees minimum brightness

    # Randomly sample hues on the color circle so that they are sufficiently distant
    # Assign the first chosen hue
    h = torch.randint(high=360, size=(1,))
    chosen_hues = chosen_hues * h

    # Sample the other hues in a way to keep sufficient angular distance
    for i in range(1, n_hues):
        # while min((abs(chosen_hues - h)) % 180) < min_hue_distance:
        while min(torch.min(abs(chosen_hues - h), torch.min(abs(chosen_hues - 360 - h), abs(chosen_hues - (h - 360))))) < min_hue_distance:
            h = torch.randint(high=360, size=(1,))
        chosen_hues[i] = h
    print("chosen hues", chosen_hues)

    # Distribute hue, saturation, value across digits
    chosen_hues = torch.repeat_interleave(chosen_hues, repeats=n_digits // n_hues)
    chosen_hues = torch.cat([chosen_hues, chosen_hues[-1].unsqueeze(0).repeat(n_digits - len(chosen_hues))])
    saturation = torch.repeat_interleave(saturation, repeats=n_digits // n_hues)
    saturation = torch.cat([saturation, saturation[-1].unsqueeze(0).repeat(n_digits - len(saturation))])
    value = torch.repeat_interleave(value, repeats=n_digits // n_hues)
    value = torch.cat([value, value[-1].unsqueeze(0).repeat(n_digits - len(value))])

    # Random permutations of hues, saturations, values per digit
    chosen_hues = chosen_hues[torch.randperm(n_digits)]
    saturation = saturation[torch.randperm(n_digits)]
    value = value[torch.randperm(n_digits)]

    # Add some other small random noise to the hues, saturations, values
    chosen_hues = ((chosen_hues + torch.rand(n_digits) * max_hue_noise) % 360)
    saturation = (torch.max(torch.zeros(n_digits), torch.min((saturation + torch.rand(n_digits) * max_saturation_noise),
                                                             torch.ones(n_digits))))
    value = (torch.max(torch.zeros(n_digits),
                       torch.min((value + torch.rand(n_digits) * max_value_noise), torch.ones(n_digits))))

    # Map HSV to RGB
    colors = torch.zeros(n_digits, 3)
    for i in range(n_digits):
        RGB = colorsys.hsv_to_rgb(chosen_hues[i] / 360, saturation[i], value[i])
        colors[i] = torch.Tensor(RGB)

    return colors


def gen_backgrounds(n=1,
                    h=200,
                    w=200,
                    n_digits=100,
                    n_hues=5,
                    datadir="backgrounds/",
                    dataset_images="MNIST",
                    digit_size=28,
                    background_color="white",
                    **args):

    print("Image size:", h, "x", w)

    # Calculate the digits coordinates
    digit_h, digit_w = digit_size, digit_size
    centers_x = np.random.randint(0+digit_h//2, high=h-digit_h//2, size=(1,n_digits), dtype=int)
    centers_y = np.random.randint(0+digit_w//2, high=w-digit_w//2, size=(1,n_digits), dtype=int)
    centers = np.concatenate((centers_x, centers_y), axis=0)

    # Define dataloader with random image rotations
    dataset = getattr(torchvision.datasets, dataset_images)
    trainset = dataset('datasets/colorMNIST/MNIST_digits/', train=True, download=True,
                       transform=torchvision.transforms.Compose([
                           torchvision.transforms.RandomRotation([0, 360]),
                           torchvision.transforms.Resize([digit_h, digit_w]),
                           torchvision.transforms.ToTensor()]))

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=n_digits, shuffle=True)

    # Create n images
    for im_count, (digits, labels) in zip(range(int(n)), train_loader):

        print(str(im_count) + '_class' + str(n_hues) + '.png')

        # Initialize image tensor
        a = torch.zeros(3, h, w)

        # Sample digit colors
        colors = sample_colors(n_digits=n_digits, n_hues=n_hues, **args)

        # Paste digit on image
        for i in range(len(centers[0])):
            x_start = centers[0][i] - digit_h // 2
            y_start = centers[1][i] - digit_w // 2
            digit = digits[i]
            digit = torch.concat((digit, digit, digit), dim=0)

            # Make non-white pixels colorful
            digit = digit.permute(1, 2, 0)

            for row in range(digit_h):
                for col in range(digit_w):
                    if not torch.equal(digit[row, col, :], torch.Tensor([0, 0, 0])):
                        digit[row, col, :] = colors[i]

            # Add color digit on image
            digit = digit.permute(2, 0, 1)
            a[:, x_start:x_start + digit_h, y_start:y_start + digit_w] = (
                torch.max(digit, a[:, x_start:x_start + digit_h, y_start:y_start + digit_w]))

        if background_color == "white":
            a = a.permute(1, 2, 0)
            zeros_tensor = torch.tensor([0, 0, 0]).view(1, 1, 3)
            is_zero = (a == zeros_tensor)
            is_zero_all_channels = is_zero.all(dim=2)
            a[is_zero_all_channels] = torch.tensor([1, 1, 1], dtype=torch.float)
            a = a.permute(2, 0, 1)

        # Save image in folder
        save_image(a, datadir + str(im_count) + '_' + str(n_hues) + 'hues.png')


def main():
    args = read_args()
    print("args", args)
    print("Creating the image folder...")
    datadir = create_data_dir(args)
    print("Generating images...")

    gen_backgrounds(n=args.n,
                    h=args.h,
                    w=args.w,
                    n_digits=args.n_digits,
                    n_hues=args.n_hues,
                    datadir=args.datadir,
                    dataset_images=args.dataset_images,
                    digit_location="random",
                    digit_size=args.digit_size,
                    background_color=args.background_color,
                    min_hue_distance=args.min_hue_distance,  # Minimum hue distance for different color classes.
                    max_hue_noise=args.max_hue_noise,  # Maximum hue variation across the same color class. Def: 3
                    saturation_range=args.saturation_range,  # Range for saturation variations. Def: [0.5, 1.0]
                    max_saturation_noise=args.max_saturation_noise,  # Maximum saturation noise. Def: 0.05
                    value_range=args.value_range,  # Range for value variations. Def: [0.5, 1.0]
                    max_value_noise=args.max_value_noise
                    )

    print("Images saved in", datadir)

    return


if __name__ == "__main__":
    main()
