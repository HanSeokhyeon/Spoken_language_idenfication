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

import os
import time
import argparse
import queue
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from loader import *
from models import CRNN
from models import CNN
from models import RNN
from util import download_data, search
from result import *

DATASET_PATH = './dataset'


def train(model, total_batch_size, queue, criterion, optimizer, device, train_begin, train_loader_count, print_batch=5):
    total_loss = 0
    total_num = 0
    total_correct = 0
    total_sent_num = 0
    batch = 0

    model.train()

    logger.info('train() start')

    begin = epoch_begin = time.time()

    while True:
        if queue.empty():
            logger.debug('queue is empty')

        feats, label, feat_lengths = queue.get()

        if feats.shape[0] == 0:
            # empty feats means closing one loader
            train_loader_count -= 1

            logger.debug('left train_loader: %d' % (train_loader_count))

            if train_loader_count == 0:
                break
            else:
                continue

        optimizer.zero_grad()

        feats = feats.to(device)
        label = label.to(device)

        logit = model(feats, feat_lengths).to(device)

        y_hat = logit.max(-1)[1]

        correct = torch.eq(y_hat, label)
        batch_correct = torch.nonzero(correct).size(0)
        total_correct += batch_correct

        loss = criterion(logit.contiguous(), label)
        total_loss += loss.item()
        total_num += logit.size(0)

        total_sent_num += label.size(0)

        loss.backward()
        optimizer.step()

        if batch % print_batch == 0:
            current = time.time()
            elapsed = current - begin
            epoch_elapsed = (current - epoch_begin) / 60.0
            train_elapsed = (current - train_begin) / 3600.0

            logger.info('batch: {:4d}/{:4d}, loss: {:.4f}, elapsed: {:.2f}s {:.2f}m {:.2f}h'
                        .format(batch,
                                total_batch_size,
                                total_loss / total_num,
                                elapsed, epoch_elapsed, train_elapsed))
            begin = time.time()

        batch += 1
        train.cumulative_batch_count += 1

    logger.info('train() completed')
    return total_loss / total_num, total_correct / total_sent_num


train.cumulative_batch_count = 0


def evaluate(model, dataloader, queue, criterion, device, confusion_matrix=None):
    logger.info('evaluate() start')
    total_loss = 0
    total_num = 0
    total_correct = 0
    total_sent_num = 0

    model.eval()

    with torch.no_grad():
        while True:
            feats, label, feat_lengths = queue.get()
            if feats.shape[0] == 0:
                break

            feats = feats.to(device)
            label = label.to(device)

            logit = model(feats, feat_lengths).to(device)

            y_hat = logit.max(-1)[1]

            if type(confusion_matrix) is torch.Tensor:
                update_confusion_matrix(confusion_matrix, label.cpu().numpy(), y_hat.cpu().numpy())

            correct = torch.eq(y_hat, label)
            batch_correct = torch.nonzero(correct).size(0)
            total_correct += batch_correct

            loss = criterion(logit.contiguous(), label)
            total_loss += loss.item()
            total_num += logit.size(0)

            total_sent_num += label.size(0)

    logger.info('evaluate() completed')
    return total_loss / total_num, total_correct / total_sent_num


def update_confusion_matrix(cm, y, y_):
    for y1, y2 in zip(y, y_):
        cm[y2][y1] += 1


def split_dataset(config, train_wav_paths, valid_wav_paths, test_wav_paths, kor_wav_paths):
    random.shuffle(kor_wav_paths)
    train_num, valid_num, test_num = len(train_wav_paths), len(valid_wav_paths), len(test_wav_paths)
    train_wav_paths = np.concatenate((train_wav_paths, kor_wav_paths[:train_num]))
    valid_wav_paths = np.concatenate((valid_wav_paths, kor_wav_paths[train_num:train_num+valid_num]))
    test_wav_paths = np.concatenate((test_wav_paths, kor_wav_paths[train_num+valid_num:train_num+valid_num+test_num]))

    random.shuffle(train_wav_paths)
    random.shuffle(valid_wav_paths)
    random.shuffle(test_wav_paths)

    train_batch_num = math.ceil(len(train_wav_paths) / config.batch_size)

    batch_num_per_train_loader = math.ceil(train_batch_num / config.workers)

    train_begin = 0
    train_end_raw_id = 0
    train_dataset_list = list()

    for i in range(config.workers-1):

        train_end = min(train_begin + batch_num_per_train_loader, train_batch_num)

        train_begin_raw_id = train_begin * config.batch_size
        train_end_raw_id = train_end * config.batch_size

        train_dataset_list.append(BaseDataset(train_wav_paths[train_begin_raw_id:train_end_raw_id], config.nn_type))

        train_begin = train_end

    train_dataset_list.append(BaseDataset(train_wav_paths[train_begin * config.batch_size:], config.nn_type))

    valid_dataset = BaseDataset(valid_wav_paths, config.nn_type)
    test_dataset = BaseDataset(test_wav_paths, config.nn_type)

    return train_batch_num, train_dataset_list, valid_dataset, test_dataset


def main():
    parser = argparse.ArgumentParser(description='Spoken Language Idenfication')
    parser.add_argument('--hidden_size', type=int, default=512, help='hidden size of model (default: 256)')
    parser.add_argument('--layer_size', type=int, default=3, help='number of layers of model (default: 3)')
    parser.add_argument('--n_class', type=int, default=2, help='number of classes of data (default: 7)')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate in training (default: 0.2')
    parser.add_argument('--bidirectional', default=True, action='store_true',
                        help='use bidirectional RNN (default: False')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size in training (default: 32')
    parser.add_argument('--workers', type=int, default=4, help='number of workers in dataset loader (default: 4)')
    parser.add_argument('--max_epochs', type=int, default=10, help='number of max epochs in training (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-04, help='learning rate (default: 0.0001)')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--save_name', type=str, default='model', help='the name of model')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--nn_type', type=str, default='crnn', help='type of neural networks')

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if args.cuda else 'cpu')

    feature_size = N_FFT / 2 + 1

    cnn = CNN.CNN(feature_size)
    rnn = RNN.RNN(cnn.feature_size, args.hidden_size, args.n_class,
                  input_dropout_p=args.dropout, dropout_p=args.dropout,
                  n_layers=args.layer_size, bidirectional=args.bidirectional, rnn_cell='gru', variable_lengths=False)

    model = CRNN.CRNN(cnn, rnn)
    model.flatten_parameters()

    model = nn.DataParallel(model).to(device)

    optimizer = optim.Adam(model.module.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(reduction='sum').to(device)

    if args.mode != 'train':
        return

    download_data()

    kor_db_list = []
    search('dataset/train/train_data', kor_db_list)

    train_wav_paths = np.loadtxt("dataset/TRAIN_list.csv", delimiter=',', dtype=np.unicode)
    valid_wav_paths = np.loadtxt("dataset/TEST_developmentset_list.csv", delimiter=',', dtype=np.unicode)
    test_wav_paths = np.loadtxt("dataset/TEST_coreset_list.csv", delimiter=',', dtype=np.unicode)

    train_wav_paths = list(map(lambda x: "dataset/TIMIT/{}.WAV".format(x), train_wav_paths))
    valid_wav_paths = list(map(lambda x: "dataset/TIMIT/{}.WAV".format(x), valid_wav_paths))
    test_wav_paths = list(map(lambda x: "dataset/TIMIT/{}.WAV".format(x), test_wav_paths))

    best_acc = 0
    begin_epoch = 0

    loss_acc = [[], [], [], []]

    train_batch_num, train_dataset_list, valid_dataset, test_dataset = \
        split_dataset(args, train_wav_paths, valid_wav_paths, test_wav_paths, kor_db_list)

    logger.info('start')

    train_begin = time.time()

    for epoch in range(begin_epoch, args.max_epochs):

        train_queue = queue.Queue(args.workers * 2)

        train_loader = MultiLoader(train_dataset_list, train_queue, args.batch_size, args.workers, args.nn_type)
        train_loader.start()

        train_loss, train_acc = train(model, train_batch_num, train_queue, criterion, optimizer, device, train_begin, args.workers, 10)
        logger.info('Epoch %d (Training) Loss %0.4f Acc %0.4f' % (epoch, train_loss,  train_acc))

        train_loader.join()

        loss_acc[0].append(train_loss)
        loss_acc[1].append(train_acc)

        valid_queue = queue.Queue(args.workers * 2)

        valid_loader = BaseDataLoader(valid_dataset, valid_queue, args.batch_size, 0, args.nn_type)
        valid_loader.start()

        eval_loss, eval_acc = evaluate(model, valid_loader, valid_queue, criterion, device)
        logger.info('Epoch %d (Evaluate) Loss %0.4f Acc %0.4f' % (epoch, eval_loss, eval_acc))

        valid_loader.join()

        loss_acc[2].append(eval_loss)
        loss_acc[3].append(eval_acc)

        best_model = (eval_acc > best_acc)

        if best_model:
            best_acc = eval_acc
            torch.save(model.state_dict(), './save_model/best_model.pt')
            save_epoch = epoch

    model.load_state_dict(torch.load('./save_model/best_model.pt'))

    test_queue = queue.Queue(args.workers * 2)

    test_loader = BaseDataLoader(test_dataset, test_queue, args.batch_size, 0, args.nn_type)
    test_loader.start()

    confusion_matrix = torch.zeros((args.n_class, args.n_class))
    test_loss, test_acc = evaluate(model, test_loader, test_queue, criterion, device, confusion_matrix)
    logger.info('Epoch %d (Test) Loss %0.4f Acc %0.4f' % (save_epoch, test_loss, test_acc))

    test_loader.join()

    save_data(loss_acc, test_loss, test_acc, confusion_matrix.to('cpu').numpy())
    plot_data(loss_acc, test_loss, test_acc)

    return 0


if __name__ == '__main__':
    main()
