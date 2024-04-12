import sys
import os
sys.path.append(os.getcwd())

import unittest
import numpy as np
import cv2 as cv
from naloga3 import kmeans, meanshift, calculate_distances, gaussian_kernel

def timeout_decorator(func):
    import functools, signal
    def _handle_timeout(signum, frame):
        raise TimeoutError()
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        signal.signal(signal.SIGALRM, _handle_timeout)
        signal.alarm(10)  # Nastavi časovno omejitev na 10 sekund
        try:
            result = func(*args, **kwargs)
        finally:
            signal.alarm(0)  # Onemogoči alarm
        return result
    return wrapper

class TestImageSegmentation(unittest.TestCase):

    def test_gaussian_kernel(self):
        distance = 5
        bandwidth = 10
        result = gaussian_kernel(distance, bandwidth)
        self.assertTrue(result < 1 and result > 0)

    def test_calculate_distances_3d(self):
        points = np.array([[0, 0, 0], [3, 4, 0]])
        expected = np.array([[0, 5], [5, 0]])
        result = calculate_distances(points, 3)
        np.testing.assert_array_almost_equal(result, expected)

    @timeout_decorator
    def test_kmeans_basic(self):
        slika = np.zeros((10, 10, 3), dtype=np.uint8)
        slika[5:, 5:] = 255
        result = kmeans(slika, k=2, iteracije=1)
        unique_colors = np.unique(result.reshape(-1, result.shape[2]), axis=0)
        self.assertTrue(len(unique_colors) >= 2)

    @timeout_decorator
    def test_meanshift_basic(self):
        slika = np.zeros((10, 10, 3), dtype=np.uint8)
        slika[5:, 5:] = 255
        result = meanshift(slika, h=10, dimenzija=5)
        unique_colors = np.unique(result.reshape(-1, result.shape[2]), axis=0)
        self.assertTrue(len(unique_colors) >= 2)

if __name__ == '__main__':
    unittest.main()

