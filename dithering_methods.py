import numpy as np
from PIL import Image


# Threshold Method
def threshold_dithering(input_image_path, output_image_path, threshold):
    # Open the image file
    image = Image.open(input_image_path)

    # Convert the image to grayscale (0.299 R + 0.587 G + 0.114 B)
    image = image.convert('L')

    # Get the width and height of the image
    width, height = image.size

    # Loop through each pixel in the image
    for y in range(height):
        for x in range(width):
            # Get the intensity value of the pixel
            intensity = image.getpixel((x, y))

            # Compare the intensity value to the threshold
            if intensity >= threshold:
                # Set the pixel to white
                image.putpixel((x, y), 255)
            else:
                # Set the pixel to black
                image.putpixel((x, y), 0)

    # Save the dithered image
    image.save(output_image_path)


def improved_threshold_dithering(input_image_path, output_image_path, threshold):
    # Open the image file and convert it to grayscale (0.299 R + 0.587 G + 0.114 B)
    image = Image.open(input_image_path).convert('L')

    # Convert the image to a NumPy array (improving performance)
    image_array = np.array(image)

    # Perform the ordered dithering using the NumPy arrays
    # Make an array of zeros with the same shape of (image_array), saving condition branch
    output_array = np.zeros_like(image_array)

    # Threshold the input image one time rather than looping through each pixel
    output_array[image_array >= threshold] = 255

    # Save the dithered image
    output_image = Image.fromarray(output_array)
    output_image.save(output_image_path)


# Floyd-Steinberg Error Diffusion Method
def floyd_steinberg(input_image_path, output_image_path):
    # Open the image file
    image = Image.open(input_image_path)

    # Convert the image to grayscale (0.299 R + 0.587 G + 0.114 B)
    image = image.convert('L')

    # Get the width and height of the image
    width, height = image.size

    # Loop through each pixel
    for y in range(height):
        for x in range(width):
            # Get the old intensity value of the pixel
            old_pixel = image.getpixel((x, y))

            # Compare to the threshold and set the new pixel value
            new_pixel = 0 if old_pixel < 128 else 255

            # Set the new pixel value
            image.putpixel((x, y), new_pixel)

            # Calculate the error
            error = old_pixel - new_pixel

            # Distribute the error over the adjacent pixels
            if x + 1 < width:
                # Get the pixel, add the error part and save the new value
                pixel = image.getpixel((x + 1, y))
                pixel += int(error * 7 / 16)
                image.putpixel((x + 1, y), pixel)
            if x + 1 < width and y + 1 < height:
                # Get the pixel, add the error part and save the new value
                pixel = image.getpixel((x + 1, y + 1))
                pixel += int(error * 1 / 16)
                image.putpixel((x + 1, y + 1), pixel)
            if y + 1 < height:
                # Get the pixel, add the error part and save the new value
                pixel = image.getpixel((x, y + 1))
                pixel += int(error * 5 / 16)
                image.putpixel((x, y + 1), pixel)
            if x > 0 and y + 1 < height:
                # Get the pixel, add the error part and save the new value
                pixel = image.getpixel((x - 1, y + 1))
                pixel += int(error * 3 / 16)
                image.putpixel((x - 1, y + 1), pixel)

    # Save the dithered image
    image.save(output_image_path)


# Ordered Dither Method
def ordered_dithering(input_image_path, output_image_path):
    # Open the image file
    image = Image.open(input_image_path)

    # Convert the image to grayscale (0.299 R + 0.587 G + 0.114 B)
    image = image.convert('L')

    # Get the width and height of the image
    width, height = image.size

    # Define the dither matrix
    dither_matrix = [(0, 128, 32, 160, 8, 136, 40, 168),
                     (192, 64, 224, 96, 200, 72, 232, 104),
                     (48, 176, 16, 144, 56, 184, 24, 152),
                     (240, 112, 208, 80, 248, 120, 216, 88),
                     (12, 140, 44, 172, 4, 132, 36, 164),
                     (204, 76, 236, 108, 196, 68, 228, 100),
                     (60, 188, 28, 156, 52, 180, 20, 148),
                     (252, 124, 220, 92, 244, 116, 212, 84)]

    # Loop through each pixel
    for y in range(height):
        for x in range(width):
            # Get the intensity value of the pixel
            pixel = image.getpixel((x, y))

            # Find corresponding dithering pixel
            dx = x % 8  # 8 is the width of "dither_matrix"
            dy = y % 8  # 8 is the width of "dither_matrix"

            # Compare image pixel with dither pixel
            if pixel > dither_matrix[dx][dy]:
                image.putpixel((x, y), 255)
            else:
                image.putpixel((x, y), 0)

    # Save the dithered image
    image.save(output_image_path)


# Ordered dithering function with NumPy arrays
def improved_ordered_dithering(input_image_path, output_image_path):
    # Open the image file and convert it to grayscale
    image = Image.open(input_image_path).convert('L')

    # Convert the image to a NumPy array (improving performance)
    image_array = np.array(image)

    # Define the dither matrix
    dither_matrix = np.array([[0, 128,  32, 160,   8, 136,  40, 168],
                              [192,  64, 224,  96, 200,  72, 232, 104],
                              [48, 176,  16, 144,  56, 184,  24, 152],
                              [240, 112, 208,  80, 248, 120, 216,  88],
                              [12, 140,  44, 172,   4, 132,  36, 164],
                              [204,  76, 236, 108, 196,  68, 228, 100],
                              [60, 188,  28, 156,  52, 180,  20, 148],
                              [252, 124, 220,  92, 244, 116, 212,  84]])

    # Perform the ordered dithering using the NumPy arrays
    # Make an array of zeros with the same shape of (image_array), saving condition branch
    output_array = np.zeros_like(image_array)

    # Loop through each pixel
    for y in range(image_array.shape[0]):
        for x in range(image_array.shape[1]):
            # Find corresponding dithering pixel
            dy = y % dither_matrix.shape[0]
            dx = x % dither_matrix.shape[1]

            # Compare the intensity value of the pixel with corresponding dither one
            if image_array[y, x] > dither_matrix[dy, dx]:
                output_array[y, x] = 255

    # Convert the output array back to an image with the original size
    output_image = Image.fromarray(output_array).resize(image.size)

    # Save the output image
    output_image.save(output_image_path)


# Generate a Bayer matrix of the specified size
def bayer_matrix(size):
    matrix = [[0, 2], [3, 1]]  # 2x2 Bayer matrix
    while len(matrix) < size:
        matrix = [x + [4 * x[-1] + i, 4 * x[-1] + 2 + i] for x in matrix for i in range(2)]
        matrix += [x + [4 * x[-1] + 3 - i, 4 * x[-1] + 1 - i] for x in matrix for i in range(2)]
    return matrix[:size][:size]


# Apply pattern dithering to the input image using the specified matrix
def pattern_dithering(input_image_path, output_image_path, matrix):
    # Open the image file and convert it to grayscale
    image = Image.open(input_image_path).convert('L')

    width, height = image.size

    output_image = Image.new('L', (width, height))
    for y in range(0, height, len(matrix)):
        for x in range(0, width, len(matrix[0])):

            block = image.crop((x, y, x + len(matrix[0]), y + len(matrix)))

            for j in range(len(matrix)):
                for i in range(len(matrix[0])):

                    threshold = matrix[j][i] / 16.0
                    output_pixel = 255 if block.getpixel((i, j)) / 255.0 > threshold else 0
                    output_image.putpixel((x + i, y + j), output_pixel)
    # Save output image
    output_image.save(output_image_path)


# Generate a Bayer matrix of the specified size
def improved_bayer_matrix(size):
    matrix = np.array([[0, 2], [3, 1]])
    while matrix.shape[0] < size or matrix.shape[1] < size:
        top = np.hstack((4 * matrix[:, -1][:, np.newaxis] + np.array([0, 1])[:, np.newaxis],
                         4 * matrix[:, -2][:, np.newaxis] + np.array([3, 2])[:, np.newaxis]))
        bottom = np.hstack((4 * matrix[:, -2][:, np.newaxis] + np.array([2, 3])[:, np.newaxis],
                            4 * matrix[:, -1][:, np.newaxis] + np.array([1, 0])[:, np.newaxis]))
        matrix = np.vstack((np.hstack((matrix, top)), np.hstack((bottom[::-1, :], np.rot90(matrix, 2)))))
    return matrix[:size, :size]


# Apply pattern dithering to the input image using the specified matrix
def improved_pattern_dithering(input_image_path, output_image_path, matrix):
    # Open the image file and convert it to grayscale
    image = Image.open(input_image_path).convert('L')

    width, height = image.size

    output_image = Image.new('L', (width, height))
    for y in range(0, height, len(matrix)):
        for x in range(0, width, len(matrix[0])):

            block = np.array(image.crop((x, y, x + len(matrix[0]), y + len(matrix)))) / 255.0

            threshold = matrix / 16.0
            output_block = np.where(block > threshold, 1.0, 0.0)

            output_image.paste(Image.fromarray(np.uint8(output_block * 255)), (x, y))

    # Save output image
    output_image.save(output_image_path)
