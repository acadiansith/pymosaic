from collections import namedtuple
from glob import iglob
import hashlib
import io
from itertools import chain
import json
import os
import sqlite3
import warnings

import progressbar

from PIL import Image
from PIL.Image import Image as PILImageClass

import numpy as np

SQLITE_COMMIT_FREQUENCY = 500
SQLITE_VECTORIZE_BATCH_SIZE = 100

MIN_PIXELS_IN_IMAGE = 20000

JPEG_QUALITY = 95

DOWNSAMPLE_PIXELS = 200000
DOWNSAMPLE_SCALE_THRESHOLD = 1  # if shape request is greater than this times what we have stored, get full image

warnings.simplefilter('error', Image.DecompressionBombWarning)


# BEGIN FROM http://stackoverflow.com/questions/18621513/python-insert-numpy-array-into-sqlite3-database
def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


# Converts np.array to TEXT when inserting
sqlite3.register_adapter(np.ndarray, adapt_array)

# Converts TEXT to np.array when selecting
sqlite3.register_converter("array", convert_array)


# END FROM http://stackoverflow.com/questions/18621513/python-insert-numpy-array-into-sqlite3-database


def adapt_jpeg(im):
    out = io.BytesIO()
    im.save(out, 'JPEG', quality=JPEG_QUALITY)
    out.seek(0)
    return out.read()

def convert_jpeg(buf):
    out = io.BytesIO(buf)
    out.seek(0)
    return Image.open(out)

def pil_image_conform(self, protocol):
    if protocol is sqlite3.PrepareProtocol:
        return adapt_jpeg(self)

PILImageClass.__conform__ = pil_image_conform

sqlite3.register_converter("jpeg", convert_jpeg)


# BEGIN FROM http://stackoverflow.com/questions/3431825/generating-an-md5-checksum-of-a-file
def hash_file(f, hasher, block_size=65536):
    buf = f.read(block_size)
    while len(buf) > 0:
        hasher.update(buf)
        buf = f.read(block_size)
    return hasher.hexdigest()


# END FROM http://stackoverflow.com/questions/3431825/generating-an-md5-checksum-of-a-file


def pil2numpy(pil):
    pil = pil if pil.mode == 'RGB' else pil.convert('RGB')
    flattened_pil = (y for x in pil.getdata() for y in x)
    return np.fromiter(flattened_pil, dtype=np.uint8).reshape((pil.size[1], pil.size[0], 3))


def open_and_crop(filename, shape):
    # NOTE: this shape is a numpy shape, not a PIL shape
    try:
        im = Image.open(filename)
        if shape is None:
            return im
        else:
            return crop_shrink(im, shape)
    except Image.DecompressionBombWarning as e:
        print(e)
        return None


def crop_to_shape(im, shape):
    # NOTE: this shape is a numpy shape, not a PIL shape
    a, b = shape
    w, h = im.size
    if w * a < h * b:
        im = im.transpose(Image.TRANSPOSE)
        w, h = im.size
        a, b = b, a
        transposed = True
    else:
        transposed = False

    out_h = h
    out_w = h * b // a
    out_x = (w - out_w) // 2
    out_y = 0

    out = im.crop((out_x, out_y, out_x + out_w, out_y + out_h))

    return out.transpose(Image.TRANSPOSE) if transposed else out


def crop_shrink(im, shape, resample=Image.BICUBIC):
    cropped = crop_to_shape(im, shape)
    return cropped.resize(shape[::-1], resample=resample)


class ImageDB:
    Config = namedtuple('Config', ('root_dir', 'cache_dir', 'file_extensions', 'shapes'))

    def __init__(self, db_filename, root_dir='.', cache_dir=None, file_extensions=('jpg', 'png'), shapes=((10, 15), (15, 10)),
                 create_db=True):
        self.db_filename = os.path.abspath(db_filename)
        if cache_dir is None:
            path, basename = os.path.split(self.db_filename)
            cache_dir = os.path.join(path, '.%s.d' % basename)
        self.config = ImageDB.Config(os.path.abspath(root_dir), os.path.abspath(cache_dir), file_extensions, shapes)

        if create_db:
            if os.path.isfile(db_filename):
                os.remove(db_filename)
            with sqlite3.connect(self.db_filename, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
                conn.execute('CREATE TABLE vectorized_images (h INTEGER, w INTEGER, count INTEGER, vectors ARRAY, image_ids ARRAY);')
                conn.execute('CREATE TABLE image_stats (h INTEGER, w INTEGER, sha256 TEXT, filename TEXT, UNIQUE(sha256));')
                conn.execute('CREATE TABLE downsampled_images (h INTEGER, w INTEGER, image_id INTEGER, filename TEXT);')
                conn.execute('CREATE TABLE config (key TEXT, val TEXT, UNIQUE(key));')

            self.save_config()
            print('Collecting image information.')
            self.collect_image_stats()
            print('Collecting downsampled images and vectorizations.')
            self.collect_downsamples_and_vectors()

    def collect_image_stats(self):
        with sqlite3.connect(self.db_filename, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
            filenames = [
                os.path.join(tup[0], fn)
                for tup in os.walk(self.config.root_dir)
                    for fn in tup[2]
                        if any(fn.endswith(ext) for ext in self.config.file_extensions)
            ]
            with progressbar.ProgressBar(max_value=len(filenames), redirect_stdout=True) as bar:
                for i, image_filename in enumerate(filenames):
                    try:
                        with Image.open(image_filename) as im:
                            w, h = im.size
                            if w * h < MIN_PIXELS_IN_IMAGE:
                                continue
                            im.verify()
                    except Exception as e:
                        print(e)
                        continue
                    with open(image_filename, 'rb') as f:
                        sha256 = hash_file(f, hashlib.sha256())
                    try:
                        conn.execute('INSERT INTO image_stats VALUES (?, ?, ?, ?)', (h, w, sha256, image_filename))
                    except sqlite3.IntegrityError:
                        print('Image %s is a duplicate. Not inserting.' % image_filename)
                    if i % SQLITE_COMMIT_FREQUENCY == 0:
                        conn.commit()
                    bar.update(i)

    def collect_downsamples_and_vectors(self):
        with sqlite3.connect(self.db_filename, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
            landscape_query = 'SELECT rowid, filename FROM image_stats WHERE h <= w;'
            portrait_query = 'SELECT rowid, filename FROM image_stats WHERE h >= w;'
            landscape_count = conn.execute('SELECT COUNT(*) FROM image_stats WHERE h <= w;').fetchone()[0]
            portrait_count = conn.execute('SELECT COUNT(*) FROM image_stats WHERE h >= w;').fetchone()[0]
            landscape_shapes = [shape for shape in self.config.shapes if shape[0] <= shape[1]]
            portrait_shapes = [shape for shape in self.config.shapes if shape[0] >= shape[1]]
            print(landscape_count, portrait_count)
            with progressbar.ProgressBar(max_value=landscape_count + portrait_count) as bar:
                subtotal = 0
                for query, count, shapes in [(landscape_query, landscape_count, landscape_shapes), (portrait_query, portrait_count, portrait_shapes)]:
                    res = conn.execute(query).fetchall()
                    for k in range(0, len(res), SQLITE_VECTORIZE_BATCH_SIZE):
                        batch = res[k:k+SQLITE_VECTORIZE_BATCH_SIZE]
                        batch_size = len(batch)
                        vector_arrays = [np.zeros((batch_size, h * w * 3)) for h, w in shapes]
                        for i, (id, filename) in enumerate(batch):
                            im = Image.open(filename)
                            for shape, vectors in zip(shapes, vector_arrays):
                                downsampled_shape_dir_name = os.path.join(self.config.cache_dir, '%dx%d' % shape)
                                os.makedirs(downsampled_shape_dir_name, exist_ok=True)
                                h, w = shape
                                cropped = crop_to_shape(im, shape)
                                downsample_scale_factor = int(np.sqrt(DOWNSAMPLE_PIXELS / h / w))
                                downsample_shape = (h*downsample_scale_factor, w*downsample_scale_factor)
                                downsampled = cropped.resize(downsample_shape[::-1], resample=Image.BICUBIC)
                                downsampled_filename = os.path.join(downsampled_shape_dir_name, '%s_%d_%d.jpg' % ((np.base_repr(id, 36).lower(),) + downsample_shape))
                                downsampled.save(downsampled_filename, quality=JPEG_QUALITY)
                                conn.execute('INSERT INTO downsampled_images VALUES (?, ?, ?, ?);', downsample_shape + (id, downsampled_filename))
                                more_downsampled = downsampled.resize(shape, resample=Image.BILINEAR)
                                vectors[i, :] = (pil2numpy(more_downsampled).astype(float) / 255).ravel()
                            bar.update(subtotal + i)
                        subtotal += batch_size

                        ids = np.array([id for id, filename in batch], dtype=int)
                        for shape, vectors in zip(shapes, vector_arrays):
                            h, w = shape
                            conn.execute('INSERT INTO vectorized_images VALUES (?, ?, ?, ?, ?)',
                                         (h, w, len(batch), vectors, ids))
                            conn.commit()

    def get_image(self, id, shape=None):
        with sqlite3.connect(self.db_filename, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
            h, w, filename = conn.execute('SELECT h, w, filename FROM image_stats WHERE rowid = ?', (id,)).fetchone()
            if filename is None:
                return None, ''
            if shape is not None:
                sh, sw = shape
                res = conn.execute('SELECT h, w, filename FROM downsampled_images WHERE image_id = ? AND h*? = w*? AND h >= ? AND w >= ? ORDER BY h ASC LIMIT 1;',
                             (id, sw, sh, sh*DOWNSAMPLE_SCALE_THRESHOLD, sw*DOWNSAMPLE_SCALE_THRESHOLD)).fetchone()
                if res is not None:
                    dh, dw, filename = res
                    im = Image.open(filename)
                    if shape == (dh, dw):
                        return im, filename
                    else:
                        return im.resize(shape[::-1], resample=Image.BICUBIC), filename
            return open_and_crop(filename, shape), filename

    def get_vectors(self, shape):
        with sqlite3.connect(self.db_filename, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
            r = conn.execute('SELECT vectors, image_ids FROM vectorized_images WHERE h = ? AND w = ?', shape).fetchall()
            if len(r) == 0:
                return None
            vectors = np.vstack(tuple(x[0] for x in r))
            ids = np.concatenate(tuple(x[1] for x in r))
        return vectors, ids

    def save_config(self):
        with sqlite3.connect(self.db_filename, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
            for k, v in self.config._asdict().items():
                conn.execute('INSERT OR IGNORE INTO config VALUES (?, ?)', (k, json.dumps(v)))
                conn.execute('UPDATE config SET val=? WHERE key=?', (json.dumps(v), k))

    def load_config(self):
        with sqlite3.connect(self.db_filename, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
            config_dict = {}
            for k in self.config._asdict():
                config_dict[k] = conn.execute('SELECT val FROM config WHERE key=?', (k,)).fetchone()[0]
            self.config = ImageDB.Config(**config_dict)

    @staticmethod
    def load(db_filename):
        image_db = ImageDB(db_filename, create_db=False)
        image_db.load_config()
        return image_db


if __name__ == '__main__':
    image_db = ImageDB.load('image_3.db')
    x, ids = image_db.get_vectors((10, 15))
    ids = [int(x) for x in ids]
    print(x.shape)
    print(len(ids))

    from skimage import io as skio

    i = 0
    print(ids[i])
    x0 = x[i, :].reshape((10, 15, 3))
    x_big, _ = image_db.get_image(ids[i])
    skio.imsave('out_vec.png', x0)
    x_big.save('out_orig.jpg', quality=95)
    x_med, fn = image_db.get_image(ids[i], (360, 540))
    print(fn)
    x_med.save('out_med.jpg', quality=95)
    x_not_so_big = x_big.resize((int(x_big.size[0]*0.5), int(x_big.size[1]*0.5)), resample=Image.BICUBIC)
    for q in [1] + list(range(5, 100, 5)):
        x_not_so_big.save('jpgtest/out_%d.jpg' % q, quality=q)

    with sqlite3.connect(':memory:', detect_types=sqlite3.PARSE_DECLTYPES) as conn:
        conn.execute('CREATE TABLE jpegs (im JPEG);')
        xbc = x_big
        conn.execute('INSERT INTO jpegs VALUES (?);', (xbc,))
        r = conn.execute('SELECT * FROM jpegs LIMIT 1;').fetchone()[0]
        r.save('out_from_db.jpg', quality=95)
