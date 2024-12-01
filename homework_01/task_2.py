import cv2
import numpy as np


def find_road_number(image: np.ndarray) -> int:
    """
    Найти номер дороги, на которой нет препятсвия в конце пути.

    :param image: исходное изображение
    :return: номер дороги, на котором нет препятсвия на дороге
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv_image, np.array([0, 120, 70]), np.array([10, 255, 255]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    h, w, _ = image.shape
    total_routes = len(contours) + 1
    route_width = w // total_routes
    
    has_obstacle = [0] * total_routes
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        path_id = x // route_width   
        if path_id < total_routes:  
            has_obstacle[path_id] = 1

    for i in range(total_routes):
        if has_obstacle[i] == 0:
            return i  
    return -1
