from typing import Tuple

import numpy as np
import cv2


def get_steer_matrix_left_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    """
    Args:
        shape:              The shape of the steer matrix.

    Return:
        steer_matrix_left:  The steering (angular rate) matrix for reactive control
                            using the masked left lane markings (numpy.ndarray)
    """
    w = shape[0]
    # TODO: implement your own solution here
    steer_matrix_left = np.ones(shape)
    #steer_matrix_left[:,0:int(np.floor(w/2))] = 0

    coef = -0.55
    steer_matrix_left *= coef
    # Set the bottom-left corner to 0 (black)
    # steer_matrix_left[corner_height:, :corner_width] = 0.5  # Mask the left-bottom corner
    # Mask the top-right corner (top right region)
     # steer_matrix_left[:corner_height, corner_width:] = 0.5  # Mask the top-right corner

    # ---
    return steer_matrix_left


def get_steer_matrix_right_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    """
    Args:
        shape:               The shape of the steer matrix.

    Return:
        steer_matrix_right:  The steering (angular rate) matrix for reactive control
                             using the masked right lane markings (numpy.ndarray)
    """

    # TODO: implement your own solution here
    w = shape[0]
    steer_matrix_right = np.ones(shape)
    #steer_matrix_right[:,int(np.floor(w/2)):w + 1] = 0

    coef = 0.38
    steer_matrix_right *= coef
    # Set the bottom-left corner to 0 (black)
    #steer_matrix_right[:corner_height, :corner_width] = -0.5  # Mask the left-bottom corner
    # Mask the top-right corner (top right region)
    #steer_matrix_right[corner_height:, corner_width:] = -0.5  # Mask the top-right corner

    # ---
    return steer_matrix_right


def detect_lane_markings(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        image: An image from the robot's camera in the BGR color space (numpy.ndarray)
    Return:
        mask_left_edge:   Masked image for the dashed-yellow line (numpy.ndarray)
        mask_right_edge:  Masked image for the solid-white line (numpy.ndarray)
    """
    h, w, _ = image.shape
    # TODO: implement your own solution here

    # Convert the image to HSV for any color-based filtering
    imghsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Most of our operations will be performed on the grayscale version
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #0. mask ground
    mask_ground = np.ones(img.shape, dtype=np.uint8)  # Start with a mask of ones (white)

    # Calculate the midpoint (half of the height)
    mid_point = img.shape[0] // 2  # Integer division to get the middle row index

    # Set the top half of the image to 0 (black)
    mask_ground[:mid_point-30, :] = 0  # Mask the top half (rows 0 to mid_point-1)

    #1. Gaussian filter
    sigma = 3.5# CHANGE ME

    # Smooth the image using a Gaussian kernel
    img_gaussian_filter = cv2.GaussianBlur(img,(0,0), sigma)

    #2 sobel and gmag
    # Convolve the image with the Sobel operator (filter) to compute the numerical derivatives in the x and y directions
    sobelx = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,1,0)
    sobely = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,0,1)

    # Compute the magnitude of the gradients
    Gmag = np.sqrt(sobelx*sobelx + sobely*sobely)

    #3 GMag treshold
    threshold = 35
    mask_mag = (Gmag > threshold)

    #4 Mask yellow and white

    white_lower_hsv = np.array([0,(0*255)/100,(60*255)/100]) # [0,0,50] - [230,100,255]
    white_upper_hsv = np.array([150,(40*255)/100,(100*255)/100])   # CHANGE ME

    yellow_lower_hsv = np.array([(30*179)/360, (30*255)/100, (30*255)/100])        # CHANGE ME
    yellow_upper_hsv = np.array([(90*179)/360, (110*255)/100, (100*255)/100])  # CHANGE ME


    mask_white = cv2.inRange(imghsv, white_lower_hsv, white_upper_hsv)
    mask_yellow = cv2.inRange(imghsv, yellow_lower_hsv, yellow_upper_hsv)


    #5 Edge masking
    # Let's create masks for the left- and right-halves of the image
    mask_left = np.ones(sobelx.shape)
    mask_left[:,int(np.floor(w/2)):w + 1] = 0
    mask_right = np.ones(sobelx.shape)
    mask_right[:,0:int(np.floor(w/2))] = 0

    mask_sobelx_pos = (sobelx > 0)
    mask_sobelx_neg = (sobelx < 0)
    mask_sobely_pos = (sobely > 0)
    mask_sobely_neg = (sobely < 0)

    #6 combine
    mask_left_edge =  mask_ground * mask_left * mask_mag * mask_sobelx_neg * mask_sobely_neg * mask_yellow
    mask_right_edge =  mask_ground * mask_right * mask_mag * mask_sobelx_pos * mask_sobely_neg * mask_white

    return mask_left_edge, mask_right_edge
