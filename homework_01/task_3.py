import cv2 as cv
import numpy as np


def rotate(image, point: tuple, angle: float) -> np.ndarray:
    """
    Повернуть изображение по часовой стрелке на угол от 0 до 360 градусов и преобразовать размер изображения.

    :param image: исходное изображение
    :param point: значение точки (x, y), вокруг которой повернуть изображение
    :param angle: угол поворота
    :return: повернутное изображение
    """
    rotation_matrix = cv.getRotationMatrix2D(point, angle, scale=1.0)
    borders = np.array([
        [0, 0],
        [0, image.shape[0] - 1],
        [image.shape[1] - 1, 0],
        [image.shape[1], image.shape[1]]
    ])
    borders_matrix = np.hstack((borders, np.ones((borders.shape[0], 1))))
    rotated = rotation_matrix @ borders_matrix.T
    
    # new borders
    rotation_matrix[:, 2] -= rotated.min(axis=1)
    rotated_image = np.int64(np.ceil(rotated.max(axis=1) - rotated.min(axis=1)))

    return cv.warpAffine(image, rotation_matrix, rotated_image)


def apply_warpAffine(image, points1, points2) -> np.ndarray:
    """
    Применить афинное преобразование согласно переходу точек points1 -> points2 и
    преобразовать размер изображения.

    :param image:
    :param points1:
    :param points2:
    :return: преобразованное изображение
    """
    affine_matrix = cv.getAffineTransform(points1, points2)
    borders = np.array([
        [0, 0],
        [0, image.shape[0] - 1],
        [image.shape[1] - 1, 0],
        [image.shape[1], image.shape[1]]
    ])
    borders_matrix = np.hstack((borders, np.ones((borders.shape[0], 1))))
    affined = affine_matrix @ borders_matrix.T
    
    # new borders
    affine_matrix[:, 2] -= affined.min(axis=1)
    affined_image = np.int64(np.ceil(affined.max(axis=1) - affined.min(axis=1)))

    return cv.warpAffine(image, affine_matrix, affined_image)
