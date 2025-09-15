"""
Demonstrating how to undistort images.

Reads in the given calibration file, parses it, and uses it to undistort the given
image. Then display both the original and undistorted images.

To use:

    python undistort.py image calibration_file
"""

import numpy as np
import cv2
import argparse
import re


class Undistorter(object):
    def __init__(self, fin: str):
        # read in distort
        with open(fin, 'r') as f:
            # chunks = f.readline().rstrip().split(' ')
            header = f.readline().rstrip()
            chunks = re.sub(r'[^0-9,]', '', header).split(',')
            self.mapu = np.zeros((int(chunks[1]), int(chunks[0])),
                                 dtype=np.float32)
            self.mapv = np.zeros((int(chunks[1]), int(chunks[0])),
                                 dtype=np.float32)
            for line in f.readlines():
                chunks = line.rstrip().split(' ')
                self.mapu[int(chunks[0]), int(chunks[1])] = float(chunks[3])
                self.mapv[int(chunks[0]), int(chunks[1])] = float(chunks[2])
        # generate a mask
        self.mask = np.ones(self.mapu.shape, dtype=np.uint8)
        self.mask = cv2.remap(self.mask, self.mapu, self.mapv, cv2.INTER_LINEAR)
        kernel = np.ones((30, 30), np.uint8)
        self.mask = cv2.erode(self.mask, kernel, iterations=1)

    def undistort(self, img: np.ndarray) -> np.ndarray:
        return cv2.resize(cv2.remap(img, self.mapu, self.mapv, cv2.INTER_LINEAR),
                          (self.mask.shape[1], self.mask.shape[0]),
                          interpolation=cv2.INTER_CUBIC)


def main():
    parser = argparse.ArgumentParser(description="Undistort images")
    parser.add_argument('image', metavar='img', type=str, help='image to undistort')
    parser.add_argument('map', metavar='map', type=str, help='undistortion map')

    args = parser.parse_args()

    undistort = Undistorter(args.map)
    print("Loaded Map.")

    im = cv2.imread(args.image)
    im_undistorted = undistort.undistort(im)
    cv2.imwrite('original.png', cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE))
    cv2.imwrite('undistorted.png', cv2.rotate(im_undistorted, cv2.ROTATE_90_CLOCKWISE))


if __name__ == "__main__":
    main()
