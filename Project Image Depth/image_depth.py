# -*- coding: utf-8 -*-
"""Image Depth.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1DVKsfl4Fc-nJKcxBKKBz5JWVItSctjHV
"""

!pip install opencv-python-headless numpy

import cv2
import numpy as np

def distance (x, y):
 return abs(x - y)

def pixel_wise_matching(left_img_path, right_img_path, disparity_range, save_result=True):
    # Read left, right images then convert to grayscale
    left = cv2.imread(left_img_path, 0)
    right = cv2.imread(right_img_path, 0)

    left = left.astype(np.float32)
    right = right.astype(np.float32)

    height, width = left.shape[:2]

    # Create blank disparity map
    depth = np.zeros((height, width), np.uint8)
    scale = 16
    max_value = 255

    for y in range(height):
        for x in range(width):
            disparity = 0
            cost_min = max_value

            for j in range(disparity_range):
                cost = max_value if (x - j) < 0 else distance(int(left[y, x]), int(right[y, x - j]))

                if cost < cost_min:
                    cost_min = cost
                    disparity = j

            # Let depth at (y, x) = j (disparity)
            # Multiply by a scale factor for visualization purpose
            depth[y, x] = disparity * scale

    if save_result == True:
        print('Saving result...')
        # Save results
        cv2.imwrite(f'pixel_wise_l1.png', depth)
        cv2.imwrite(f'pixel_wise_l1_color.png', cv2.applyColorMap(depth, cv2.COLORMAP_JET))

    print('Done.')
    return depth

# Assuming you have uploaded your images to Google Colab
left_img_path = '/content/left.png'  # Replace with your actual image path
right_img_path = '/content/right.png'  # Replace with your actual image path
disparity_range = 16

disparity_map = pixel_wise_matching(left_img_path, right_img_path, disparity_range)

## Problem 2 ##
def distance(x, y):
    return abs(x - y)

def window_based_matching(left_img_path, right_img_path, disparity_range, kernel_size=5, save_result=True):
    # Read left, right images then convert to grayscale
    left = cv2.imread(left_img_path, 0)
    right = cv2.imread(right_img_path, 0)

    left = left.astype(np.float32)
    right = right.astype(np.float32)

    height, width = left.shape[:2]

    # Create blank disparity map
    depth = np.zeros((height, width), np.uint8)

    kernel_half = int((kernel_size - 1) / 2)
    scale = 3
    max_value = 255 * 9

    for y in range(kernel_half, height - kernel_half):
        for x in range(kernel_half, width - kernel_half):
            disparity = 0
            cost_min = 65534

            for j in range(disparity_range):
                total = 0

                for v in range(-kernel_half, kernel_half + 1):
                    for u in range(-kernel_half, kernel_half + 1):
                        value = max_value
                        if (x + u - j) >= 0:
                            value = distance(int(left[y + v, x + u]), int(right[y + v, (x + u) - j]))
                        total += value

                if total < cost_min:
                    cost_min = total
                    disparity = j

            # Let depth at (y, x) = j (disparity)
            # Multiply by a scale factor for visualization purpose
            depth[y, x] = disparity * scale

    if save_result:
        print('Saving result...')
        # Save results
        cv2.imwrite(f'window_based_l1.png', depth)
        cv2.imwrite(f'window_based_l1_color.png', cv2.applyColorMap(depth, cv2.COLORMAP_JET))

    print('Done.')
    return depth

left_img_path = '/content/Aloe_left_1.png'
right_img_path = '/content/Aloe_right_1.png'
disparity_range = 64
kernel_size = 5

disparity_map = window_based_matching(left_img_path, right_img_path, disparity_range, kernel_size)

from matplotlib import pyplot as plt

plt.imshow(disparity_map, cmap='gray')
plt.title('Disparity Map (Window-Based Matching)')
plt.show()

##problem 3 ##

left_img_path = '/content/Aloe_left_1.png'
right_img_path = '/content/Aloe_right_2.png'
disparity_range = 64
kernel_size = 5


disparity_map = window_based_matching(left_img_path, right_img_path, disparity_range, kernel_size)


from matplotlib import pyplot as plt

plt.imshow(disparity_map, cmap='gray')
plt.title('Disparity Map with Disparity Range 64 and Kernel Size 5')
plt.show()