import numpy as np

from PIL import Image
from skimage import io, transform

import tiling
from tiling import Tiling


def image_from_tiles(image_shape, boxes, image_ids):
    h, w = image_shape
    im = Image.new('RGB', (w, h))
    n = boxes.shape[0]

    for k in range(n):
        box = boxes[k, :]
        image_id = image_ids[k]
        if np.any(box):
            tile = get_image(image_id, tuple(box[2:]))
            im.paste(tile, (box[1], box[0], box[1]+box[3], box[0]+box[2]))

    return im


def get_image(image_id, shape):
    return Image.fromarray(np.uint8(patches[image_id, :].reshape(shape + (3,))*255))


def greedy_assignment(x, y, batch_size=10):
    ni = x.shape[0]
    nj = y.shape[0]

    x2 = (x**2).sum(1).reshape((ni, 1))
    y2 = (y**2).sum(1).reshape((1, nj))

    assignment = np.zeros(ni, dtype=int)
    
    unused = np.ones(nj)

    search_order = np.arange(ni)
    np.random.shuffle(search_order)
    
    for k in range(0, ni, batch_size):
        if k % (10*batch_size) == 0:
            print(k)
        dists = -2*x[search_order[k:k+batch_size], :].dot(y.T) + x2[search_order[k:k+batch_size]] + y2
        max_dist = dists.max()
        dists = (max_dist - dists)*unused.reshape((1, nj))
        need_assignment = np.ones(dists.shape[0], dtype=bool)
        while sum(need_assignment) > 0:
            js = np.zeros_like(need_assignment, dtype=int)
            js[need_assignment] = dists[need_assignment, :].argmax(1)
            for i, j in enumerate(js):
                if need_assignment[i] and unused[j]:
                    assignment[search_order[k+i]] = j
                    unused[j] = False
                    dists[:, j] = 0
                    need_assignment[i] = False

    return assignment


n_samples = 20000 # number of random tiles
n_tiles = 5000 # number of tiles to include in mosaic

pattern = tiling.parquet_3x2_2p_1l_pattern
tiling = Tiling(tiling_pattern=pattern, scale=5)
tile_shape = tiling.get_scaled_tile_shapes()[0][0]

print('Generating random tiles.')
patches = np.zeros((n_samples, tile_shape[0]*tile_shape[1]*3))
for k in range(n_samples):
    patches[k, :] = (np.random.rand(*(tile_shape + (3,)))*0.3 + np.random.rand(1, 1, 3)*0.7).ravel()

x = io.imread('hellokitty.jpg').astype(float)/255
x = transform.rescale(x, np.sqrt(n_tiles*tile_shape[0]*tile_shape[1]/x.shape[0]/x.shape[1]))

print('Extracting image patches.')
x_vs, x_bs = tiling.get_tile_patch_vectors(x)
x_v = np.vstack(x_vs)
x_b = np.vstack(x_bs)

print('Finding assignment.')
image_ids = greedy_assignment(x_v, patches, 50)

print('Constructing mosaic.')
im_mosaic = image_from_tiles(x.shape[:2], x_b, image_ids)

im_mosaic.save('hk_mosaic.png')

