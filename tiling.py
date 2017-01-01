import math
from collections import namedtuple

from PIL import Image

import numpy as np

TilingPattern = namedtuple('TilingPattern', ['name', 'tile_shapes', 'shape', 'overlap', 'tiles'])

Tile = namedtuple('Tile', ['i', 'j', 'shape'])

single_square_pattern = TilingPattern(
    name='single_square',
    tile_shapes=[((1, 1), 1)],
    shape=(1, 1),
    overlap=(0, 0),
    tiles=[
        Tile(i=0, j=0, shape=0)
    ]
)

parquet_3x2_2p_1l_pattern = TilingPattern(
    name='parquet_3x2_2p1_l1',
    tile_shapes=[((3, 2), 18), ((2, 3), 9)],
    shape=(18, 9),
    overlap=(2, 2),
    tiles=[
        Tile(i=0, j=0, shape=1),  # 0
        Tile(i=-1, j=3, shape=0),
        Tile(i=-2, j=5, shape=0),
        Tile(i=0, j=7, shape=0),
        Tile(i=1, j=5, shape=0),
        Tile(i=2, j=0, shape=0),  # 5
        Tile(i=2, j=2, shape=1),
        Tile(i=3, j=7, shape=0),
        Tile(i=4, j=2, shape=0),
        Tile(i=4, j=4, shape=1),
        Tile(i=5, j=0, shape=0),  # 10
        Tile(i=6, j=4, shape=0),
        Tile(i=6, j=6, shape=1),
        Tile(i=7, j=2, shape=0),
        Tile(i=8, j=-1, shape=1),
        Tile(i=8, j=6, shape=0),  # 15
        Tile(i=9, j=4, shape=0),
        Tile(i=10, j=-1, shape=0),
        Tile(i=10, j=1, shape=1),
        Tile(i=11, j=6, shape=0),
        Tile(i=12, j=1, shape=0),  # 20
        Tile(i=12, j=3, shape=1),
        Tile(i=13, j=-1, shape=0),
        Tile(i=14, j=3, shape=0),
        Tile(i=14, j=5, shape=1),
        Tile(i=15, j=1, shape=0),  # 25
        Tile(i=16, j=-2, shape=1),
    ]
)


class Tiling:

    def __init__(self, tiling_pattern=single_square_pattern, scale=1):
        self.tiling_pattern = tiling_pattern
        self.scale = scale

    def get_tile_patch_vectors(self, x):
        ni, nj, _ = x.shape
        mi, mj = self.tiling_pattern.shape
        mi *= self.scale
        mj *= self.scale
        oi, oj = self.tiling_pattern.overlap
        oi *= self.scale
        oj *= self.scale

        pi = math.ceil((ni + oi) / mi)
        pj = math.ceil((nj + oj) / mj)
        patch_vectors = [np.zeros((pi*pj*k, qi*qj*3*self.scale**2)) for (qi, qj), k in self.tiling_pattern.tile_shapes]
        patch_boxes = [np.zeros((pi*pj*k, 4), dtype=int) for _, k in self.tiling_pattern.tile_shapes]
        patch_counts = [0]*len(self.tiling_pattern.tile_shapes)
        for ti in range(pi):
            for tj in range(pj):
                for tk, tile in enumerate(self.tiling_pattern.tiles):

                    i = ti*mi + tile.i*self.scale
                    j = tj*mj + tile.j*self.scale
                    h, w = self.tiling_pattern.tile_shapes[tile.shape][0]
                    h *= self.scale
                    w *= self.scale

                    if i >= ni or j >= nj:
                        continue

                    patch = np.zeros((h, w, 3))
                    patch_filled = False

                    # pad patches that hang off the edge
                    if i < 0:
                        patch[-i:, :min(w, nj-j), :] = x[0:i+h, j:j+w, :]
                        for k in range(-i):
                            patch[k, :, :] = patch[-i, :, :]
                        patch_filled = True
                    if j < 0:
                        if not patch_filled:
                            patch[:min(h, ni-i), -j:, :] = x[i:i+h, 0:j+w, :]
                        for k in range(-j):
                            patch[:, k, :] = patch[:, -j, :]
                        patch_filled = True
                    if i+h > ni:
                        if not patch_filled:
                            patch[:ni-i, :min(w, nj-j), :] = x[i:, j:j+w, :]
                        for k in range(ni-i, h):
                            patch[k, :, :] = patch[k-1, :, :]
                        patch_filled = True
                    if j+w > nj:
                        if not patch_filled:
                            patch[:min(h, ni-i), :nj-j, :] = x[i:i+h, j:, :]
                        for k in range(nj-j, w):
                            patch[:, k, :] = patch[:, k-1, :]
                        patch_filled = True
                    if not patch_filled:
                        patch[:, :, :] = x[i:i+h, j:j+w, :]

                    patch_vectors[tile.shape][patch_counts[tile.shape], :] = patch.ravel()
                    patch_boxes[tile.shape][patch_counts[tile.shape], :] = np.array([i, j, h, w], dtype=int)
                    patch_counts[tile.shape] += 1

        for k in range(len(self.tiling_pattern.tile_shapes)):
            patch_vectors[k] = patch_vectors[k][:patch_counts[k], :]
            patch_boxes[k] = patch_boxes[k][:patch_counts[k], :]

        return patch_vectors, patch_boxes

    def get_scaled_tile_shapes(self):
        return [((qi*self.scale, qj*self.scale), k) for ((qi, qj), k) in self.tiling_pattern.tile_shapes]


def crop_to_shape(im, shape):
    # this uses wxh shapes, rather than rowxcol shapes
    a, b = shape
    w, h = im.size
    if w*b < h*a:
        im = im.transpose(Image.TRANSPOSE)
        w, h = im.size
        a, b = b, a
        transposed = True
    else:
        transposed = False

    out_h = h
    out_w = h*a // b
    out_x = (w - out_w) // 2
    out_y = 0

    out = im.crop((out_x, out_y, out_x + out_w, out_y + out_h))

    return out.transpose(Image.TRANSPOSE) if transposed else out


if __name__ == '__main__':
    from skimage import io
    import numpy as np

    pattern = parquet_3x2_2p_1l_pattern
    x = np.zeros((pattern.shape[0] + pattern.overlap[0], pattern.shape[1] + pattern.overlap[1], 3))
    for i, tile in enumerate(pattern.tiles):
        i = tile.i + pattern.overlap[0]
        j = tile.j + pattern.overlap[1]
        x[i:i + pattern.tile_shapes[tile.shape][0][0], j:j + pattern.tile_shapes[tile.shape][0][1], :] = np.random.rand(3)
    io.imsave('test_pattern.png', x)

    x = np.zeros((10, 10, 3))
    x[:, :, 0] = np.arange(100).reshape((10, 10))
    x[:, :, 1] = x[:, :, 0]
    x[:, :, 2] = x[:, :, 0]

    tiling = Tiling(tiling_pattern=parquet_3x2_2p_1l_pattern, scale=3)
    print(tiling.get_tile_patch_vectors(x))

