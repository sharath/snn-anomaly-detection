import numpy as np
import pandas as pd
import torch
from scipy.misc import imsave
from scipy.sparse import coo_matrix


def centroid(data, l, u):
    return int(sum(data['lat'][l:u]) / (u - l)), int(sum(data['long'][l:u]) / (u - l))


def draw_dot(matrix, pos, value):
    x, y = pos
    depth = 255 - 25 * value

    def softset(x, y):
        try:
            matrix[x][y] = depth
        except IndexError:
            pass

    softset(x, y)

    return matrix


def crop(sparse, center, width):
    cropped = np.zeros((width, width))

    corner = int(center[0] - (width / 2)), int(center[1] - (width / 2))
    for i in range(width):
        for j in range(width):
            if corner[0] + i in sparse.row and corner[1] + j in sparse.col:
                idx = list(sparse.row).index(corner[0] + i)
                if sparse.col[idx] == corner[1] + j:
                    cropped = draw_dot(cropped, (i, j), sparse.data[idx])
                else:
                    idx = list(sparse.col).index(corner[1] + j)
                    if sparse.row[idx] == corner[0] + i:
                        cropped = draw_dot(cropped, (i, j), sparse.data[idx])
    return cropped


def encode(inpt, resolution, width, consider=5, save=False):
    global count
    # processing data
    data = pd.read_csv(inpt, header=None).drop(columns=[4, 6, 7, 8])
    data.columns = ['route', 'timestamp', 'long', 'lat', 'velocity']
    data['timestamp'] = (data['timestamp'] * 1e-3).astype(int)
    data['lat'] = ((data['lat'] + 90) * resolution).astype(int)
    data['long'] = ((data['long'] + 180) * resolution).astype(int)

    ret = torch.empty((0, width, width))
    for start in range(len(data) - consider):
        sparse = coo_matrix((360 * resolution, 180 * resolution), dtype=np.int)
        for order, (lat, long) in enumerate(
                zip(data['lat'][start:start + consider], data['long'][start:start + consider])):
            sparse.row = np.append(sparse.row, lat)
            sparse.col = np.append(sparse.col, long)
            sparse.data = np.append(sparse.data, order)

        center = centroid(data, start, start + consider)
        count = 0
        cropped = crop(sparse, center, width)
        if save:
            imsave('%d.png' % start, cropped)
        cropped = torch.from_numpy(cropped).view((1, width, width)).float()
        ret = torch.cat((ret, cropped))
    return ret


if __name__ == '__main__':

    try:
        import sys

        assert len(sys.argv) == 2
    except AssertionError:
        print('Specify Input File.')
        exit(-1)

    try:
        import os

        os.stat(sys.argv[1])
    except FileNotFoundError:
        print('Input File Does Not Exist.')
        exit(-1)
    else:
        encode(sys.argv[1], int(1e5), 100, save=True)
