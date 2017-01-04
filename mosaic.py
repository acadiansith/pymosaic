import argparse
import csv

import progressbar

import numpy as np

from PIL import Image
from skimage import io, transform

import tiling
from tiling import Tiling

import imagedb
from imagedb import ImageDB


class MosaicGenerator:

    def __init__(self, image_db, tiling=Tiling(tiling_pattern=tiling.parquet_3x2_2p_1l_pattern, scale=5)):
        self.tiling = tiling
        self.image_db = image_db


    def create_mosaic(self, im, n_tiles=1000, output_pixels=1000000):
        if isinstance(im, str):
            im = imagedb.open_and_crop(im)
        if not Image.isImageType(im):
            raise ValueError('Image must be a filename or a valid PIL image.')

        w_in, h_in = im.size

        tile_shapes = self.tiling.get_scaled_tile_shapes()
        average_pixels_per_tile = sum(qh*qw*k for (qh, qw), k in tile_shapes) / sum(k for _, k in tile_shapes)
        in_scale_factor = np.sqrt(n_tiles*average_pixels_per_tile/w_in/h_in)
        w, h = int(w_in*in_scale_factor), int(h_in*in_scale_factor)

        x = imagedb.pil2numpy(im.resize((w, h), resample=Image.BICUBIC))

        print('Extracting image patches.')
        x_vs, x_bs = self.tiling.get_tile_patch_vectors(x)
        boxes = np.vstack(x_bs)

        print('Determining mosaic assignment.')
        image_ids = []
        box_count = len(boxes)
        with progressbar.ProgressBar(max_value=box_count) as bar:
            subtotal = 0
            def update_bar(k):
                bar.update(subtotal + k)
            for shape, x_v in zip((shape for shape, k in tile_shapes), x_vs):
                image_db_vectors, image_db_ids = self.image_db.get_vectors(shape)
                assignment = greedy_assignment(x_v, image_db_vectors, 50, update_bar=update_bar)
                image_ids.append(image_db_ids[assignment])
                subtotal += len(assignment)
        image_ids = np.concatenate(image_ids)

        print('Constructing mosaic.')
        out_scale_factor = int(np.sqrt(output_pixels / w / h))
        w_out, h_out = int(w*out_scale_factor), int(h*out_scale_factor)
        out_boxes = boxes*out_scale_factor
        im_mosaic, filenames = self.image_from_tiles((h_out, w_out), out_boxes, image_ids)

        return im_mosaic, out_boxes, filenames


    def image_from_tiles(self, image_shape, boxes, image_ids):
        h, w = image_shape
        im = Image.new('RGB', (w, h))
        n = boxes.shape[0]
        filenames = []

        with progressbar.ProgressBar(max_value=n) as bar:
            for k in range(n):
                box = [int(x) for x in boxes[k, :]]
                image_id = int(image_ids[k])
                if np.any(box):
                    tile, filename = self.image_db.get_image(image_id, box[2:])
                    im.paste(tile, (box[1], box[0], box[1]+box[3], box[0]+box[2]))
                else:
                    filename = ''
                filenames.append(filename)
                bar.update(k)

        return im, filenames


def greedy_assignment(x, y, batch_size=50, update_bar=None):
    ni = x.shape[0]
    nj = y.shape[0]

    x2 = (x**2).sum(1).reshape((ni, 1))
    y2 = (y**2).sum(1).reshape((1, nj))

    assignment = np.zeros(ni, dtype=int)
    
    unused = np.ones(nj)

    search_order = np.arange(ni)
    np.random.shuffle(search_order)
    
    for k in range(0, ni, batch_size):
        if update_bar is not None:
            update_bar(k)
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

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create image mosaics.')
    parser.add_argument('image_db_filename', metavar='db', type=str, help='image database filename')
    parser.add_argument('input_filename', metavar='in', type=str, help='input image filename')
    parser.add_argument('output_prefix', metavar='out', type=str, help='output image prefix')
    parser.add_argument('--quality', metavar='out', type=int, default=95, help='output image quality')
    parser.add_argument('--n_tiles', type=int, default=10000, help='number of tiles')
    parser.add_argument('--n_pixels', type=int, default=20000000, help='number of pixels in mosaic')

    args = parser.parse_args()

    idb = ImageDB.load(args.image_db_filename)
    m = MosaicGenerator(idb)
    im = Image.open(args.input_filename)
    mosaic, bs, filenames = m.create_mosaic(im, n_tiles=args.n_tiles, output_pixels=args.n_pixels)

    out_filename = '%s.jpg' % args.output_prefix
    print('Saving mosaic to %s' % out_filename)
    mosaic.save(out_filename, quality=args.quality)

    csv_filename = '%s_tiles.csv' % args.output_prefix
    print('Saving mosiac tile info to %s' % csv_filename)
    with open(csv_filename, 'w', newline='') as f:
        csvwriter = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(['i', 'j', 'h', 'w', 'filename'])
        for b, filename in zip(bs, filenames):
            csvwriter.writerow(list(b) + [filename])



# n_samples = 20000 # number of random tiles
# n_tiles = 5000 # number of tiles to include in mosaic
#
# pattern = tiling.parquet_3x2_2p_1l_pattern
# tiling = Tiling(tiling_pattern=pattern, scale=5)
# tile_shape = tiling.get_scaled_tile_shapes()[0][0]
#
# print('Generating random tiles.')
# patches = np.zeros((n_samples, tile_shape[0]*tile_shape[1]*3))
# for k in range(n_samples):
#     patches[k, :] = (np.random.rand(*(tile_shape + (3,)))*0.3 + np.random.rand(1, 1, 3)*0.7).ravel()
#
# x = io.imread('hellokitty.jpg').astype(float)/255
# x = transform.rescale(x, np.sqrt(n_tiles*tile_shape[0]*tile_shape[1]/x.shape[0]/x.shape[1]))
#
# print('Extracting image patches.')
# x_vs, x_bs = tiling.get_tile_patch_vectors(x)
# x_v = np.vstack(x_vs)
# x_b = np.vstack(x_bs)
#
# print('Finding assignment.')
# image_ids = greedy_assignment(x_v, patches, 50)
#
# print('Constructing mosaic.')
# im_mosaic = image_from_tiles(x.shape[:2], x_b, image_ids)
#
# im_mosaic.save('hk_mosaic.png')

