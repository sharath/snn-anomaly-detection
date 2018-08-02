import hashlib
import itertools
import math
import sys

import numpy as np
import pandas as pd
from nupic.bindings.math import Random
from pyproj import Proj, transform


class GeospatialCoordinateEncoder:
    def __init__(self,
                 scale=5,
                 timestep=10,
                 w=21,
                 n=1000):
        self.w = w
        self.n = n
        self.scale = scale
        self.timestep = timestep

    def _orderForCoordinate(self, coordinate):
        seed = self._hashCoordinate(coordinate)
        rng = Random(seed)
        return rng.getReal64()

    def _topWCoordinates(self, coordinates, w):
        orders = np.array([self._orderForCoordinate(c) for c in coordinates.tolist()])
        indices = np.argsort(orders)[-w:]
        return coordinates[indices]

    def encodeIntoArray(self, inputData, output):
        altitude = None
        if len(inputData) == 4:
            (speed, longitude, latitude, altitude) = inputData
        else:
            (speed, longitude, latitude) = inputData
        coordinate = self.coordinateForPosition(longitude, latitude, altitude)
        radius = self.radiusForSpeed(speed)

        assert isinstance(radius, int), ("Expected integer radius, got: {} ({})"
                                         .format(radius, type(radius)))

        neighbors = self._neighbors(coordinate, radius)
        winners = self._topWCoordinates(neighbors, self.w)

        bitFn = lambda coordinate: self._bitForCoordinate(coordinate, self.n)
        indices = np.array([bitFn(w) for w in winners])
        print(len(indices))

    #        output[:] = 0
    #       output[indices] = 1

    def _neighbors(self, coordinate, radius):
        ranges = (xrange(n - radius, n + radius + 1) for n in coordinate.tolist())
        return np.array(list(itertools.product(*ranges)))

    def _hashCoordinate(self, coordinate):
        coordinateStr = ",".join(str(v) for v in coordinate)
        hash = int(int(hashlib.md5(coordinateStr).hexdigest(), 16) % (2 ** 64))
        return hash

    def _bitForCoordinate(self, coordinate, n):
        seed = self._hashCoordinate(coordinate)
        rng = Random(seed)
        return rng.getUInt32(n)

    def coordinateForPosition(self, longitude, latitude, altitude=None):
        PROJ = Proj(init="epsg:3785")  # Spherical Mercator
        geocentric = Proj('+proj=geocent +datum=WGS84 +units=m +no_defs')
        coords = PROJ(longitude, latitude)

        if altitude is not None:
            coords = transform(PROJ, geocentric, coords[0], coords[1], altitude)

        coordinate = np.array(coords)
        coordinate = coordinate / self.scale
        return coordinate.astype(int)

    def radiusForSpeed(self, speed):
        overlap = 1.5
        coordinatesPerTimestep = speed * self.timestep / self.scale
        radius = int(round(float(coordinatesPerTimestep) / 2 * overlap))
        minRadius = int(math.ceil((math.sqrt(self.w) - 1) / 2))
        return max(radius, minRadius)


if __name__ == '__main__':
    inpt = sys.argv[1]
    enc = GeospatialCoordinateEncoder()
    data = pd.read_csv(inpt, header=None).drop(columns=[4, 6, 7, 8])
    data.columns = ['route', 'timestamp', 'long', 'lat', 'velocity']
    data['timestamp'] = (data['timestamp'] * 1e-3).astype(int)

    inputData = (data['velocity'].tolist(), data['long'].tolist(), data['lat'].tolist())
    output = np.zeros(5)
    enc.encodeIntoArray(inputData, output)
