import cv2
import numpy as np
from collections import deque


STEPS = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
]


def find_way_from_maze(image: np.ndarray) -> tuple:
    """
    Найти путь через лабиринт.

    :param image: изображение лабиринта
    :return: координаты пути из лабиринта в виде (x, y), где x и y - это массивы координат
    """
    # transfer to image with 0 and 1
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

    h, w = binary_image.shape

    # find first white cell (begin of the maze) and last white cell (end)
    begin = (0, np.where(binary_image[0] == 255)[0][0])
    end = (h - 1, np.where(binary_image[-1] == 255)[0][0])

    # bfs to find the path
    q = deque([begin])
    visited = set([begin])
    prev = dict()
    prev[begin] = None

    while len(q) > 0:
        cur = q.popleft()
        if cur == end:
            break

        for step in STEPS:
            next = (cur[0] + step[0], cur[1] + step[1])

            if (0 <= next[0] < h) and (0 <= next[1] < w) and binary_image[next] == 255 and next not in visited:
                q.append(next)
                visited.add(next)
                prev[next] = cur

    # build the path by bfs results
    path = []
    step = end
    while step:
        path.append(step)
        step = prev[step]
    
    x_path, y_path = zip(*path)
    return np.array(x_path), np.array(y_path)
