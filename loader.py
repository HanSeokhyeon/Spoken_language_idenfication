"""
Copyright 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

#-*- coding: utf-8 -*-

import sys
import math
import wavio
import torch
import random
import threading
import logging
import numpy as np
import librosa
from torch.utils.data import Dataset
from torchaudio.transforms import MFCC

logger = logging.getLogger('root')
FORMAT = "[%(asctime)s %(filename)s:%(lineno)s - %(funcName)s()] %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
logger.setLevel(logging.INFO)
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib').disabled = True

N_FFT = 512
SAMPLE_RATE = 16000


def get_spectrogram_feature(filepath):
    if filepath.split('/')[1] == 'TIMIT':
        sig = np.fromfile(filepath, dtype=np.int16)[512:].reshape((-1, 1))
    else:
        (fate, width, sig) = wavio.readwav(filepath)

    sig = sig.ravel().astype(np.float) / 32767
    sig = librosa.resample(sig, 16000, 8000)
    mfcc = librosa.feature.mfcc(sig,
                                sr=8000,
                                n_fft=512,
                                hop_length=256,
                                n_mfcc=40,
                                center=False)
    mfcc = np.mean(mfcc, axis=1)

    return torch.FloatTensor(mfcc)


def get_label(filepath):
    label = filepath.split('/')[1]

    label2num = {"TIMIT": 0, "train": 1}

    return label2num[label]


class BaseDataset(Dataset):
    def __init__(self, wav_paths):
        self.wav_paths = wav_paths

    def __len__(self):
        return len(self.wav_paths)

    def count(self):
        return len(self.wav_paths)

    def getitem(self, idx):
        feat = get_spectrogram_feature(self.wav_paths[idx])
        label = get_label(self.wav_paths[idx])
        return feat, label


def _collate_fn(batch):
    feat_size = batch[0][0].size(0)
    batch_size = len(batch)

    seqs = torch.zeros(batch_size, feat_size)

    targets = torch.zeros(batch_size).to(torch.long)

    for x in range(batch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seqs[x].narrow(0, 0, feat_size).copy_(tensor)
        target = torch.LongTensor([target])
        targets.narrow(0, x, 1).copy_(target)

    return seqs, targets


class BaseDataLoader(threading.Thread):
    def __init__(self, dataset, queue, batch_size, thread_id):
        threading.Thread.__init__(self)
        self.collate_fn = _collate_fn
        self.dataset = dataset
        self.queue = queue
        self.index = 0
        self.batch_size = batch_size
        self.dataset_count = dataset.count()
        self.thread_id = thread_id

    def count(self):
        return math.ceil(self.dataset_count / self.batch_size)

    def create_empty_batch(self):
        seqs = torch.zeros(0, 0)
        target = torch.zeros(0).to(torch.long)
        return seqs, target

    def run(self):
        logger.debug('loader %d start' % (self.thread_id))
        while True:
            items = list()

            for i in range(self.batch_size):
                if self.index >= self.dataset_count:
                    break

                items.append(self.dataset.getitem(self.index))
                self.index += 1

            if len(items) == 0:
                batch = self.create_empty_batch()
                self.queue.put(batch)
                break

            random.shuffle(items)

            batch = self.collate_fn(items)
            self.queue.put(batch)
        logger.debug('loader %d stop' % (self.thread_id))


class MultiLoader():
    def __init__(self, dataset_list, queue, batch_size, worker_size):
        self.dataest_list = dataset_list
        self.queue = queue
        self.batch_size = batch_size
        self.worker_size = worker_size
        self.loader = list()

        for i in range(self.worker_size):
            self.loader.append(BaseDataLoader(self.dataest_list[i], self.queue, self.batch_size, i))

    def start(self):
        for i in range(self.worker_size):
            self.loader[i].start()

    def join(self):
        for i in range(self.worker_size):
            self.loader[i].join()
