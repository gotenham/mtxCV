# mtxCV
Matrix based Computer Vision for low resolution microcontroller applications; based on Eigen matrix and vector library.

There are many existing robust open source computer vision libraries available such as OpenCV and others, these libraries are generally heavy in terms of resource requirements in the context of microcontrollers (STM32, Arduino etc); therefore this project aims to port basic image processing and edge detection functionality to lower end hardware by leaveraging the matrix math and manipulation functions available in the Eigen library. 

Currently supported inputs (single channel greyscale):
- 2D float / int pixel intensity matrix
- 1D float / int pixel intensity array

Currently supported processes:
- Generate NxN gaussian kernel
- Decompose kernel matrix using SVD to equivelant row and col kernel vectors
- Convolution using 2D kernel matrix
- Convolution using seperable row and col kernel vectors
- Histogram equalisation
- Normalise image
- Suppress pixel intensity below ratio of intensity range
- Canny Edge Detection
- Dynamic calculation of Canny threasholds using Otsu's method
- Non-Maximal Gradient Suppression with interpolation between neighbouring pixels
- Edge Tracking by Hysteresis using Breadth First Search
- Get 2nd Order Geometric Moments for image patch

Currently supported outputs:
- Print image to serial using ASCII gradients
- Print image intensity to serial in table format

## Example ASCII Serial Output
My personal implementation targets STM32 ARM chips, but a x64 build of example_mtxCV.cpp is provided for illustration.
VS code image output as serial ASCII gradient using example image from example_mtxCV.cpp is below:

Example image of earth from orbit:

![image](https://github.com/gotenham/mtxCV/assets/40827722/427bfbb2-e571-43fa-b1fe-801bb407736f)

Image processing with 5x5 gaussian blur:

![image](https://github.com/gotenham/mtxCV/assets/40827722/b6cbdb20-3c38-4edf-89e6-73fb13b0a3e3)

Canny candidate edge mask:

![image](https://github.com/gotenham/mtxCV/assets/40827722/4cedce19-16df-42bb-8a93-b0566f0e01e9)

Identified edge sequence:

![image](https://github.com/gotenham/mtxCV/assets/40827722/5b78da2f-b67c-4ea3-8c4c-a97697bd4966)

## Example Serial Output to GUI Interface Device
The images can also be sent to serial UART in standardised format (written similar to CSV); this can be interpreted by a python or equivelant script running the OpenCV library on the receiving device to better visualise the results.

Example of mtxCV processed image and candidate edge mask running on STM32 L4 microcontroller from live thermal image feed, displayed on receiving device using openCV:

![image](https://github.com/gotenham/mtxCV/assets/40827722/9957b45d-a537-4726-8315-bffca05cf9b0)
![image](https://github.com/gotenham/mtxCV/assets/40827722/bb7a11eb-c7f5-4179-b2ec-548dc8c3dbdb)

![image](https://github.com/gotenham/mtxCV/assets/40827722/244d7b15-01b5-4cbe-b271-576c726b359e)
![image](https://github.com/gotenham/mtxCV/assets/40827722/915226b2-f6b4-40cb-afa7-19eba86d6390)

![image](https://github.com/gotenham/mtxCV/assets/40827722/5c71200a-690c-4346-8fda-875d17f69381)
![image](https://github.com/gotenham/mtxCV/assets/40827722/8d6c55a5-f4b7-45ef-ac38-c796a099ad39)

## Next steps
This is a work in progress, contributors and feedback welcome; on the TBD list:
- Remove dependancies on dynamic memory usage and convert to fixed size compilation (eg alternitives for vector, queue, unordered_map etc)
- Implement Zernike Moments for subpixel edge correction
- General optimisations and improvements
- etc...
