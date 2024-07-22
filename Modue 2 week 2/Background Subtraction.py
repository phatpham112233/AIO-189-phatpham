import cv2
import numpy as np

def compute_difference(bg_img, input_img):
    difference = cv2.absdiff(bg_img, input_img)
    gray_difference = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    return gray_difference

def compute_binary_mask(difference_single_channel):
    _, binary_mask = cv2.threshold(difference_single_channel, 50, 255, cv2.THRESH_BINARY)
    return binary_mask

def replace_background(bg1_image, bg2_image, ob_image):
    difference_single_channel = compute_difference(bg1_image, ob_image)
    binary_mask = compute_binary_mask(difference_single_channel)
    output = np.where(binary_mask[:, :, None] == 255, ob_image, bg2_image)
    return output


bg1_image = cv2.imread('C:\Users\Admin\AIO-189-phatpham-1\GreenBackground.png')
bg2_image = cv2.imread('C:\Users\Admin\AIO-189-phatpham-1\NewBackground.jpg')
ob_image = cv2.imread('C:\Users\Admin\AIO-189-phatpham-1\Object.png')

bg1_image = cv2.resize(bg1_image, (678, 381))
bg2_image = cv2.resize(bg2_image, (678, 381))
ob_image = cv2.resize(ob_image, (678, 381))

result = replace_background(bg1_image, bg2_image, ob_image)
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
