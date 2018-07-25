import sys

import numpy as np
import pandas as pd
from scipy.misc import imsave
from scipy.sparse import coo_matrix


def centroid(data, l, u):
    return int(sum(data['lat'][l:u]) / (u - l)), int(sum(data['long'][l:u]) / (u - l))


def draw_dot(matrix, pos, value, lim):
    x, y = pos
    depth = max(255 - (25 * (lim - value)), 0)

    def softset(x, y):
        try:
            matrix[x][y] = depth
        except IndexError:
            pass

    softset(x - 1, y)
    softset(x + 1, y)
    softset(x, y + 1)
    softset(x, y - 1)
    softset(x, y)

    return matrix


def crop(sparse, center, width, order_lim):
    cropped = np.zeros((width, width))

    corner = int(center[0] - (width / 2)), int(center[1] - (width / 2))
    for i in range(width):
        for j in range(width):
            if corner[0] + i in sparse.row and corner[1] + j in sparse.col:
                idx = list(sparse.row).index(corner[0] + i)
                if sparse.col[idx] == corner[1] + j and sparse.data[idx] < order_lim:
                    cropped = draw_dot(cropped, (i, j), sparse.data[idx], order_lim)
                else:
                    idx = list(sparse.col).index(corner[1] + j)
                    if sparse.row[idx] == corner[0] + i and sparse.data[idx] < order_lim:
                        cropped = draw_dot(cropped, (i, j), sparse.data[idx], order_lim)
    return cropped


def main(inpt, resolution, width, consider=5):
    # processing data
    data = pd.read_csv(inpt, header=None).drop(columns=[4, 6, 7, 8])
    data.columns = ['route', 'timestamp', 'long', 'lat', 'velocity']
    data['timestamp'] = (data['timestamp'] * 1e-3).astype(int)
    data['lat'] = ((data['lat'] + 90) * resolution).astype(int)
    data['long'] = ((data['long'] + 180) * resolution).astype(int)

    sparse = coo_matrix((360 * resolution, 180 * resolution), dtype=np.int)
    for order, (lat, long) in enumerate(zip(data['lat'], data['long'])):
        sparse.row = np.append(sparse.row, lat)
        sparse.col = np.append(sparse.col, long)
        sparse.data = np.append(sparse.data, order)

    for start in range(len(data) - consider):
        center = centroid(data, start, start + consider)
        cropped = crop(sparse, center, width, start + consider)
        print('Creating %d.png' % start)
        imsave('%d.png' % start, cropped)

    print('Done.')


if __name__ == '__main__':
    try:
        assert len(sys.argv) == 2
    except AssertionError:
        print('Specify Input File.')
    else:
        main(sys.argv[1], int(1e5), 100)
