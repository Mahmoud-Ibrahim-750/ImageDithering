# Image Dithering

Image Dithering is a Python package that provides a simple implementation of several popular dithering algorithms, including Threshold, Error diffusion, Ordered and Pattern (Halftone) Dithering techniques. Dithering is a technique used in image processing to reduce the number of colors in an image while still maintaining the appearance of the original image.

## Installation

To install Image Dithering, simply use pip:

```pip install imagedithering```



## Usage

Here's an example of how to use Image Dithering to convert an image to a 4-bit color palette using the Floyd-Steinberg algorithm:

```
from imagedithering import dither

image_path = 'path/to/image.png'
palette_size = 4
algorithm = 'floyd-steinberg'

dithered_image = dither(image_path, palette_size, algorithm)
dithered_image.save('path/to/dithered_image.png')
```

## Contributing

If you would like to contribute to Image Dithering, please follow these steps:

1- Fork the repository
2- Create a new branch (git checkout -b feature/foo)
3- Make your changes and commit them (git commit -am 'Add some feature')
4- Push to the branch (git push origin feature/foo)
5- Create a new pull request



# Image Dithering

This repository contains code for image dithering, a technique used to convert an image from a high number of colors to a lower number of colors while still preserving its visual appearance. The code is written in Python and uses the NumPy and Pillow libraries.

## Installation

To use this code, you need to have Python installed on your system. You can download Python from the official website: https://www.python.org/downloads/  Once you have Python installed, you can install the required libraries by running the following command:

```
pip install numpy pillow
```

## Usage

To use the image dithering code, you need to run the `dithering.py` file. The file takes two arguments: the path to the input image and the number of colors you want to reduce the image to. For example, to dither the image `input.png` to 16 colors, run the following command:

```
dithering.py input.png 16
```
The output image will be saved as `output.png` in the same directory as the input image.  

## Examples

Here are some examples of images before and after dithering:  ![Original Image](examples/lena.png)  ![Dithered Image (16 colors)](examples/lena_16.png)  ![Dithered Image (8 colors)](examples/lena_8.png)

## License
This code is licensed under the MIT License. See the LICENSE file for details. 
