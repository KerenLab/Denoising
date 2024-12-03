import numpy as np
import cv2
import torch

def remove_aggregations(fov, min_area=50):
    """
    Given an input FOV of shape (c, h, w), this function cleans the image
    from little objects in each channel separately.

    Args:
        fov (numpy.ndarray): Input FOV of shape (c, h, w)
        min_area (int): Minimum area of objects to be kept (default: 50)

    Returns:
        numpy.ndarray: Cleaned FOV of shape (c, h, w)
    """
    cleaned_fov = np.zeros_like(fov)
    fov_avg_pool = (torch.nn.AvgPool2d(kernel_size=5, stride=1, padding=2)(torch.from_numpy(fov)) > 0).numpy()

    for c in range(fov_avg_pool.shape[0]):
        channel = fov_avg_pool[c]
        channel = channel.astype(np.uint8)
        _, thresh = cv2.threshold(channel, 0, 255, cv2.THRESH_BINARY) # + cv2.THRESH_OTSU

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                cv2.drawContours(cleaned_fov[c], [contour], 0, 1, -1)
    
    cleaning_mask = (cleaned_fov > 0).astype(float)

    return fov * cleaning_mask