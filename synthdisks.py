"""
Create augmented synthetic images with random disks to use for YOLO
object detection training with custom data.
Each image file is saved with a corresponding text file containing the
bounding box and label data in YOLO format (class, x-center, y-center, h, w).
Uncomment the img_shape variable (listed below local imports) for the
desired image size; or edit the img_shape variable as needed.
If running from an IDE, edit main() to set the desired number of images
and their starting file name index. The default is num_image_files = 10
and start_idx = 0.
If running from the command line, uncomment in main():
    if len(argv) > 1:
        start_idx = manage.arguments()["start_idx"]
        num_image_files = manage.arguments()["num_files"]
...and comment out:
    num_image_files = 10
    start_idx = 0
"""
# Copyright (C) 2024 C.S. Echt, under GNU General Public License

# Standard imports.
from pathlib import Path
from random import choices, randint, uniform
from math import sqrt

# Third-party imports.
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from skimage import io as skio
from skimage.draw import disk
from skimage.filters import gaussian
from skimage.transform import resize
from skimage.util import random_noise

# Local imports
from synth_utils import manage

# img_shape = (3024, 4032, 3)  # 4k, iPhone 4:3 photo size, landscape orientation.
# img_shape = (4032, 3024, 3)  # 4k, iPhone 4:3 photo size, portrait orientation.
img_shape = (640, 640, 3)  # for YOLO training.
# img_shape = (1280, 960, 3)  # for YOLO training.
# img_shape = (960, 1280, 3)  # for YOLO training.
# img_shape = (960, 960, 3)  # for YOLO training.


def fill_with_color(shape: tuple[int, int, int]) -> tuple[np.ndarray, tuple[int, int, int]]:
    """
    Make an array of size *shape* with one random color or gray.
    The returned is used only in set_mottling_shading().
    :param shape: The shape of the image to create.
    :return: A numpy array with a random color or gray and the RGB value.
    """
    image = np.zeros(shape=shape, dtype=np.uint8)

    # Fill the empty image with a random RGB color, including ~20% grays.
    # 0-40 is dark range; 225-255 is light range.
    lo_val = 0
    hi_val = 255
    if randint(1, 5) == 1:
        gray_value = randint(20, 235)
        rgb = (gray_value, gray_value, gray_value)
    else:
        rgb = (randint(lo_val, hi_val),
               randint(lo_val, hi_val),
               randint(lo_val, hi_val))

    image[:, :, :] = rgb
    return image, rgb


def make_all_noisy(img2noise: np.ndarray) -> np.ndarray:
    """
    Add random noise to an entire image.
    :param img2noise: The output of draw_disks().
    :return: A numpy unit8 array with a noisy image.
    """

    # see: https://scikit-image.org/docs/dev/api/skimage.util.html#skimage.util.random_noise
    # In my opinion, 'poisson' is the most realistic noise for photo images,
    #  and 's&p' mimics specks of light reflection and dirt.
    random_mode = choices(['poisson', 's&p'])[0]

    if random_mode in 'poisson, localvar':
        noisy_img = random_noise(image=np.asarray(img2noise),
                                 mode=random_mode)
    elif random_mode in 's&p, salt, pepper':
        noisy_img = random_noise(image=np.asarray(img2noise),
                                 mode=random_mode,
                                 amount=0.05)  #  0.1 too much; 0.05 is good.
    else:  # is 'gaussian' or 'speckle'
        noisy_img = random_noise(image=np.asarray(img2noise),
                                 mode=random_mode,
                                 mean=0.10,
                                 var=0.05)

    return np.asarray((noisy_img * 255), dtype=np.uint8)


def make_bkg_noisy(shape: tuple[int, int, int]) -> np.ndarray:
    """
    Create a noisy image with a coarse grid pattern.
    A chance of mimicking real-world backgrounds.
    :param shape: The shape of the image to create.
    :return: A numpy array with a noisy image.
    """
    noisy = np.random.randint(0, 255, size=shape, dtype=np.uint8)

    # Randomize level of noise coarseness, less coarse better for small images.
    # 1=full resolution, 2=half, 3=third.
    # coarseness = choices(population=(1, 2, 3), k=1)[0]
    coarseness = choices(population=(1, 2), k=1)[0]

    # idea from: https://stackoverflow.com/questions/18666014/downsample-array-in-python
    down_sampled = noisy[::coarseness, ::coarseness]

    # Resize back to original shape to make a coarser grid pattern.
    # Mode options: 'constant', 'edge', 'wrap', 'reflect', 'symmetric'
    #  constant=fill with 0s; edge=fill with edge values; wrap=wrap around, symmetric=reflect.
    # 'edge' is like 'wrap' but with a single pixel border. Works well for noise.
    # Anti-aliasing: True=smooth the image; False=keep blocky.
    # Interpolation order options: 0, 1, 2, 3, 4, 5;
    #  0=nearest (default), 1=bi-linear, 3=cubic, 5=quintic. Higher=slower.
    return resize(image=down_sampled,
                  output_shape=shape,
                  mode='edge',
                  anti_aliasing=False,
                  order=0,
                  preserve_range=True,  # Keep original values.
                  ).astype(np.uint8)


def set_mottling_shading(rbg_value: tuple[int, int, int], index: int) -> tuple[int, int, int]:
    """
    Generate random RGB values for mottling.
    Called from the make_bkg_mottled() ellipse drawing loop.
    :param rbg_value: The RGB value to adjust.
    :param index: The index of the ellipse being drawn.
    :return: An RGB tuple with values lightened or darkened from input.
    """
    _R, _G, _B = rbg_value
    # Need factors to produce a narrow range of color shade differences
    #  between circles and from the background color.
    #  The narrower the ranges, the more subtle the shade variation.
    # shade_factor = uniform(0.45, 0.55) # 0.5 is no change.
    lighten_factor = uniform(1.1, 1.3)
    darken_factor = uniform(0.7, 0.9)

    # Adjust shading based on even/odd circle index number.
    if index % 2 == 0:
        # Lighten the color
        r = _R * lighten_factor
        g = _G * lighten_factor
        b = _B * lighten_factor
    else:
        # Darken the color
        r = _R * darken_factor
        g = _G * darken_factor
        b = _B * darken_factor

    return int(r), int(g), int(b)


def make_bkg_mottled(shape: tuple[int, int, int]) -> np.ndarray:
    """
    Create a mottled background image.
    Called from make_image().
    Calls set_mottling_shading() and fill_with _color().
    :param shape: The shape of the image to create; expecting img_shape
    :return: A numpy array of a randomly mottled image.
    """
    # Sources: https://stackoverflow.com/questions/59991178/
    #  https://stackoverflow.com/questions/59991178/creating-pixel-noise-with-pil-python
    #      creating-pixel-noise-with-pil-python
    # A smaller mottle_grain gives a finer grained mottle.
    # mottle_grain is also used to set blur radius in the returned image.
    # Note that varied mottling effects can be achieved by changing the
    #   num_in_area and mottle_grain values. For example, a num_in_area
    #   that's 1/4th of the square root of the image area, and a mottle_grain
    #   that's 1/10th of the num_in_area, will give finer mottle effect that
    #   emphasizes light and dark variations from the base color. A num_in_area
    #   that's 1/2 of the square root of the area and 1/20th of the number_in_area
    #   gives a more gentle mottling emphasizing the darker shades.
    #   1/3rd and 1/20th works well on large images.
    #   1/6th and 1/10th finer grain with good light/dark balance on large images.
    #   1/10 and 1/2 gives a less dense, well-balanced, mottle across a range
    #    of image sizes. This is a good compromise. For a 4034x3024 4k image,
    #    the BoxBlur radius is 174 (mottle_grain), and num_in_area is 349.

    # _w and _h are the width and height of the image and not the
    #   r x c shape of the ndarray.
    img_w, img_h = shape[0], shape[1]
    colored_img, rgb = fill_with_color(shape=shape)
    num_in_area = int(sqrt(img_w * img_h)) // 10
    mottle_grain = num_in_area // 2

    # Convert ndarray to PIL image for ellipse drawing.
    colored_img = Image.fromarray(colored_img)
    draw = ImageDraw.Draw(colored_img)

    # Use lots of blured ellipses to create a mottled background.
    for i in range(num_in_area):
        # Positions of the ellipse in image.
        pos_w = randint(mottle_grain, img_w - mottle_grain)
        pos_h = randint(mottle_grain, img_h - mottle_grain)

        # Size of Bounding Box for ellipse, as a function of image size.
        el_x = randint(mottle_grain,
                       pos_w // img_w + mottle_grain)
        el_y = randint(mottle_grain,
                       pos_h // img_h + mottle_grain)

        # Increasing el_y length makes the mottling more horizontal.
        # Increasing el_x length makes the mottling more vertical.
        # Randomly switching them makes the cumulative effect more chaotic.
        if randint(1, 2) == 1:
            el_y *= 2
        else:
            el_x *= 2

        x0 = pos_w - el_x
        y0 = pos_h - el_y
        x1 = pos_w + el_x
        y1 = pos_h + el_y

        top_left = (y0, x0)
        bottom_right = (y1, x1)
        bbox = (top_left, bottom_right)

        # Set the color and draw an ellipse.
        color = set_mottling_shading(rgb, i)
        draw.ellipse(xy=bbox, fill=color)

    # Blur the ellipses.
    result = colored_img.filter(ImageFilter.BoxBlur(radius=mottle_grain))

    return np.asarray(result, dtype=np.uint8)


def blur_image(img2blur: np.ndarray) -> np.ndarray:
    """
    Blur an entire image.
    :param img2blur: The output of draw_disks().
    :return: A numpy unit8 array with a blurred image.
    """

    # A check for dev/debugging in case img2blur is 1D, not 2D or 3D.  .
    # if img2blur.ndim < 2:
    #     print('Insufficient dimensions to blur; skipping blur step.')
    #     return img2blur
    trunc = sqrt(img_shape[0] * img_shape[1]) / 1000
    blurry_img = gaussian(img2blur,
                          sigma=(5, 5),
                          truncate=trunc,
                          channel_axis=-1)

    return np.asarray((blurry_img * 255), dtype=np.uint8)


def make_image() -> np.ndarray:
    """
    Make an image with a random background. The background can be
    noise, random color, or a mottled random color.
    """

    img_bkg = (
        # np.full(shape=img_shape, fill_value=255, dtype=np.uint8), # white
        # np.full(shape=img_shape, fill_value=0, dtype=np.uint8), # black
        make_bkg_noisy(shape=img_shape),
        fill_with_color(shape=img_shape)[0],
        make_bkg_mottled(shape=img_shape)
    )

    prefs = (0.15, 0.35, 0.50)
    # prefs = (0.0, 1, 0.0)  # TESTING: Use only solid background.
    # prefs = (0.0, 0.0, 1.0)  # TESTING: Use only mottled background.
    # prefs = (1, 0.0, 0)  # TESTING: Use only noise background.

    return choices(population=img_bkg, weights=prefs)[0]


# def yolo_formatter(center: tuple, radius: int) -> str:
def yolo_formatter(data: list) -> str:

    """
    Convert disk drawing px coordinates to a YOLO format:
    <class> <x_center> <y_center> <width> <height>
    :param data: A list of disk's center pixel coordinates and radius.
    :return: A string with the YOLO formatted data to use as labels.
        The class is always 0, as there is only one class.
    """
    if data:
        center = data[:2]
        radius = data[-1]
        x_center = f'{(center[1] / img_shape[1]):6f}'
        y_center = f'{(center[0] / img_shape[0]):6f}'
        width = f'{(radius * 2 / img_shape[1]):6f}'
        height = f'{(radius * 2 / img_shape[0]):6f}'
        return f"0 {x_center} {y_center} {width} {height}"
    else: # No disk data; used for background images.
        return ""


def filter_disks(disk_data: list[tuple]) -> set:
    """
    Measure the distance between two points.
    :param disk_data: List of tuples of disk center coordinates and
        their radii.
    :return: The distance between the two points.
    """
    # Source: https://stackoverflow.com/questions/51509808/
    #  how-to-measure-distances-in-a-image-with-python

    filtered_disk_data = []
    # Check for disk occlusion. NOTE: The first disks in the list are
    # the last disks drawn. Any given disk[i] will NOT be occluded by
    # a subsequent disk[j]. So filter out any disk[j] that is occluded.
    # Occlusion is when the distance between the centers of two disks
    # is less than the sum of their radii; i.e., they overlap.
    for i in range(len(disk_data)):
        center1 = disk_data[i][:2]
        r1 = disk_data[i][-1]
        #  Need to include solo disks; no need for distances
        if len(disk_data) == 1:
            filtered_disk_data.append(disk_data[0])
            break
        for j in range(i+1, len(disk_data)):
            center2 = disk_data[j][:2]
            r2= disk_data[j][-1]
            (x1, y1) = center1
            (x2, y2) = center2
            center_distance = sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            # Allow for a 2% overlap to account for rounding errors.
            if r1 + r2 <= int(center_distance * 1.02):
                filtered_disk_data.append(disk_data[i])

            # Need to append the last disk in the list (the last drawn),
            #  which will always be topmost. When disk count is 2, the
            #  last disk is always topmost.
            if j == len(disk_data) - 1:
                filtered_disk_data.append(disk_data[j])

    return set(filtered_disk_data)


def adjust_contrast(image: np.ndarray, rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    """
    Adjust the brightness of color contrast between background and disks.
    Adjusts the RGB values of the disks to ensure they are visible against
    the background. The function checks the standard deviation of the
    background colors. If the standard deviation is low, the background
    is likely a solid color. If the standard deviation is high, the
    background is likely random. If the standard deviation is in a range
    between 2 and 25, the background is likely mottled. The function
    then adjusts the luminosity (brightness) values of the disks to ensure
    they are detectable against the background.
    :param image: A numpy array of the image to adjust, passed from make_image().
    :param rgb: The RGB color to adjust.
    :return: A tuple with suitable RGB values.
    """
    std_bkg = np.std(image, ddof=0, axis=(0, 1))
    mean_std = np.mean(std_bkg)
    bkg_color = (0, 0, 0)
    if mean_std in (72, 73, 74):  # 73 expected, with 127 or 127 rgb mean
        bkg_type = 'random'
    elif mean_std == 0:
        bkg_type = 'solid'
        # bkg_color = np.unique(image.reshape(-1, 3), axis=0)[0]
        bkg_color = np.average(image, axis=(0, 1)).astype(np.uint8)
    else:  # mottled, mean_std is likely in range(2,25)
        bkg_type = 'mottled'
        bkg_color = np.average(image, axis=(0, 1)).astype(np.uint8)

    # Always draw disks that have a useful color difference from the background.
    # Source: https://stackoverflow.com/questions/9018016/
    #  how-to-compare-two-colors-for-similarity-difference
    # https://www.nbdtech.com /Blog/archive/2008/04/27/
    #   Calculating-the-Perceived-Brightness-of-a-Color.aspx
    rgb2 = rgb
    if bkg_type in 'solid, mottled':
        r1, g1, b1 = bkg_color
        r2, g2, b2 = rgb2
        brightness1 = ((.241 * r1 ** 2) + (.691 * g1 ** 2) + (.068 * b1 ** 2)) ** 0.5
        brightness2 = ((.241 * r2 ** 2) + (.691 * g2 ** 2) + (.068 * b2 ** 2)) ** 0.5
        delta_b = abs(brightness2 - brightness1)
        # Adjust color value differences.
        # delta_c = sqrt(abs(r2 - r1) ^ 2 + abs(g2 - g1) ^ 2 + abs(b2 - b1) ^ 2)
        # p = delta / sqrt(255 ^ 2 + 255 ^ 2 + 255 ^ 2)
        # while p < 0.25:
        #     rgb2 = randint(0, 255), randint(0, 255), randint(0, 255)
        #     r2, g2, b2 = rgb2
        #     delta_c = sqrt(abs(r2 - r1) ^ 2 + abs(g2 - g1) ^ 2 + abs(b2 - b1) ^ 2)
        #     p = delta_c / sqrt(255 ^ 2 + 255 ^ 2 + 255 ^ 2)
        #     print(f'Bkg color:{bkg_color}, Old disk color:{rgb}, NEW disk color:{rgb2}')

        # 35 is a good delta_b threshold for disk-background contrast; >=40 is strong.
        while delta_b < 35:
            rgb2 = randint(0, 255), randint(0, 255), randint(0, 255)
            r2, g2, b2 = rgb2
            brightness2 = ((.241 * r2 ** 2) + (.691 * g2 ** 2) + (.068 * b2 ** 2)) ** 0.5
            delta_b = abs(brightness2 - brightness1)
            # print(f'Delta_b now: {int(delta_b)}, brightness changed from {int(brightness1)} to'
            #       f' {int(brightness2)}')

    # Return the color with a suitable brightness contrast.
    return rgb2


def draw_disks() -> tuple[np.ndarray, list]:
    """
    Draw random disks on image array background.
    Calls make_image() to get the background.
    :return: A numpy array with random disks drawn on it.
    """
    # Use copy to avoid drawing over the original image.
    made_image = make_image().copy()

    # Disk radius range is ~4% - 15% of image shortest side.
    # Need to limit disks to always be within the image boundary.
    radius_max = int(min(img_shape[0], img_shape[1]) * 0.15)
    radius_min = radius_max // 4
    center_max = min(img_shape[0], img_shape[1]) - radius_max
    center_min = radius_max

    # Need a factor to limit positions of disks to image aspect.
    aspect_px_diff = abs(img_shape[0] - img_shape[1])

    # Randomize number, position, size, and color of disks.
    # Draw disks across the entire image area, regardless of aspect, but
    #  try to keep them from touching an image edge.
    # Have ~50% of disks be a random gray, the rest random colors.
    # Check for similar colors between disk and background. If too similar,
    #  change to a random color.
    yolo_data = []
    centers = []
    # for _ in range(randint(1, 4)):
    # Limit to 2 to ensure occluded disks are filtered out.
    # Need to work on occlusion filtering for multiple disks.
    for i in range(randint(1, 2)):
        if img_shape[0] < img_shape[1]:  # is portrait
            center = (
                randint(center_min, center_max),
                randint(center_min, center_max + aspect_px_diff)
                )
        else:  # is landscape or square
            center = (
                randint(center_min, center_max + aspect_px_diff),
                randint(center_min, center_max)
                )

        radius = randint(radius_min, radius_max)

        if randint(1, 2) % 2 == 0:
            x = randint(0, 255)
            rgb = (x, x, x)
        else:
            rgb = randint(0, 255), randint(0, 255), randint(0, 255)

        # Check color contrasts between disks and background.
        # Returned rgb is the adjusted luminosity, if needed.
        rgb = adjust_contrast(image=made_image, rgb=rgb)

        # Draw the disk.
        rr, cc = disk(center=center, radius=radius, shape=made_image.shape)
        made_image[rr, cc] = rgb

        centers.append(center + (radius,))

    # Filter out occluded disks. Currently only works for 2 disks.
    filtered_disks: set = filter_disks(centers)
    for disk_data in filtered_disks:
        yolo_data.append(yolo_formatter(disk_data))

    return np.asarray(made_image, dtype=np.uint8), yolo_data


def main():
    """
    Loop to create, transform, and save specified number of images and
    the YOLO bounding box data for disks in each image.
    """

    path_to_saved_images = 'saved_images'
    # path_to_saved_images = '../viz_yolo_labels/bboxviz_data'
    Path(path_to_saved_images).mkdir(exist_ok=True)  # Create a folder, if needed.

    # Start file name index at start_idx. You can avoid overwriting existing
    #  images if start_idx is larger than last saved file number in the destination
    #  folder, path_to_saved_images.

    # If running from the command line, for example, to get 30 images starting
    #  at number 150,use: python3 -m synth_disks -s 150 -n 30
    #  and uncomment the if statement below.
    # When invoked as `$ python3 -m synth_disks`, (1 argument) this program will
    #  use default values to start index 0 and create 25 images. Default values
    #  are set in synth_utils/manage.arguments.py.

    # if len(argv) > 1:
    #     start_idx = manage.arguments()["start_idx"]
    #     num_image_files = manage.arguments()["num_files"]

    # If running from within an IDE, just edit start and num values below.
    #  Comment out these two variables when running from the command line.
    num_image_files = 10
    start_idx = 0

    print(f"Creating {num_image_files} images of size {img_shape[1]} x {img_shape[0]} pixels.")

    # Process and save the images.
    # Notes on image transformation probabilities, based on random choices
    #  set below and in make_image().
    # The probabilities approximate:
    # ~50% of backgrounds and discs are grayscale.
    # ~20% of images (every 5th) are blurred.
    # ~25% have noise applied the image, and 15% start with noisy background.
    # 50% of all images have a mottled background.
    # Will get more realistic results if apply noise before blur,
    #  instead of after blur; so, blur the noise.
    for i in range(start_idx, num_image_files + start_idx):
        imgfile = Path(f"{path_to_saved_images}/disk_image_{i}.jpg")
        txtfile = Path(f"{path_to_saved_images}/disk_image_{i}.txt")
        # imgfile = Path(f"{path_to_saved_images}/disk_bkg_{i}.jpg")
        # txtfile = Path(f"{path_to_saved_images}/disk_bkg_{i}.txt")

        # yolo_data = ['']  # Empty string for background images.
        # img = make_image().copy()  # Background image only.
        img, yolo_data = draw_disks()
        if randint(1, 4) == 1:
            img = make_all_noisy(img)
        if i % 5 == 0:
            img = blur_image(img)

        # The img variable here is expected to be a ndarray. If you pass images as
        #  PIL.Image, then import imageio and save with: imageio.imwrite(uri=imgfile, im=img).
        skio.imsave(fname=imgfile, arr=img, check_contrast=False)
        print(f"Saved {imgfile}")

        # Save the file's disk YOLO data in a text file.
        with open(txtfile, 'w') as f:
            for data in yolo_data:
                f.write(f"{data}\n")
        print(f"Saved disk data in YOLO format as {txtfile}")


if __name__ == '__main__':
    manage.arguments()
    main()
