import os
import numpy as np
import shutil, requests, zipfile, io
from logger import *


def download_data():
    if os.path.isdir('dataset/TIMIT'):
        logger.info("TIMIT already exists")
    else:
        logger.info("TIMIT downloading")
        r = requests.get('https://ndownloader.figshare.com/files/10256148')
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall('dataset')
        shutil.move('dataset/data/lisa/data/timit/raw/TIMIT', 'dataset')
        shutil.rmtree('dataset/data')

    if os.path.isdir('dataset/train'):
        logger.info("Korean DB already exists")
    else:
        logger.info("Korean DB downloading")
        r = requests.get('')


def search(dirname, input_filename):
    try:
        filenames = os.listdir(dirname)
        for filename in filenames:
            full_filename = os.path.join(dirname, filename)
            if os.path.isdir(full_filename):
                search(full_filename, input_filename)
            else:
                ext = os.path.splitext(full_filename)[-1]
                if ext == '.PHN':
                    input_filename.append(full_filename[:-4])
                    # print(full_filename)
    except PermissionError:
        pass


def get_data(filename):
    wav_filename = filename + ".WAV"

    y = np.fromfile(wav_filename, dtype=np.int16)
    y = (y[512:] + 0.5) / 32767.5

    return y


def get_delta(x, N):
    pad_x = np.pad(x, ((0, 0), (N, N)), 'edge')
    delta = np.zeros(np.shape(x))
    iterator = [i + 1 for i in range(N)]
    for t in range(np.shape(x)[1]):
        tmp1, tmp2 = 0, 0
        for n in iterator:
            tmp1 += n * (pad_x[:, (t + N) + n] - pad_x[:, (t + N) - n])
            tmp2 += 2 * n * n
        delta[:, t] = np.divide(tmp1, tmp2)

    return delta


def normalize_data(x, data_mean, data_std):
    data = x
    data_std[data_std==0] = 0.00001
    data[:-1] -= data_mean[:-1, None]
    data[:-1] /= data_std[:-1, None]
    return data
