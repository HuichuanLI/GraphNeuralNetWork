# -*- coding:utf-8 -*-
# @Time : 2021/9/12 12:11 下午
# @Author : huichuan LI
# @File : run_eges.py
# @Software: PyCharm


import pandas as pd
import numpy as np
import tensorflow as tf
import time
import argparse

from eges import EGES_Model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--n_sampled", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--root_path", type=str, default='./data_cache/')
    parser.add_argument("--num_feat", type=int, default=4)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--outputEmbedFile", type=str, default='./embedding/EGES.embed')
    args = parser.parse_args()

    # read train_data
    print('read features...')
    start_time = time.time()
    side_info = np.loadtxt(args.root_path + 'sku_side_info.csv', dtype=np.int32, delimiter='\t')
    feature_lens = []
    for i in range(side_info.shape[1]):
        tmp_len = len(set(side_info[:, i]))
        feature_lens.append(tmp_len)
    end_time = time.time()
    print('time consumed for read features: %.2f' % (end_time - start_time))


    # read data_pair by tf.dataset
    def decode_data_pair(line):
        columns = tf.strings.split([line], ' ')
        x = tf.strings.to_number(columns.values[0], out_type=tf.int32)
        y = tf.strings.to_number(columns.values[1], out_type=tf.int32)
        return x, y


    dataset = tf.data.TextLineDataset(args.root_path + 'all_pairs').map(decode_data_pair,
                                                                        num_parallel_calls=tf.data.AUTOTUNE).prefetch(
        500000)
    # dataset = dataset.shuffle(256)
    dataset = dataset.repeat(args.epochs)
    dataset = dataset.batch(args.batch_size)  # Batch size to use
    iterator = tf.compat.v1.data.make_one_shot_iterator(
        dataset
    )

    print('read embedding...')
    start_time = time.time()
    EGES = EGES_Model(len(side_info), args.num_feat, feature_lens,
                      n_sampled=args.n_sampled, embedding_dim=args.embedding_dim, lr=args.lr)
    end_time = time.time()
    print('time consumed for read embedding: %.2f' % (end_time - start_time))
    opt = tf.keras.optimizers.Adam(0.01)
    print_every_k_iterations = 100
    iteration = 0
    start = time.time()
    while iterator:
        iteration += 1

        batch_index, batch_labels = iterator.get_next()
        with tf.GradientTape() as tape:
            loss = EGES(side_info, batch_index, batch_labels)
        gradients = tape.gradient(loss, EGES.trainable_variables)
        opt.apply_gradients(zip(gradients, EGES.trainable_variables))
        # 计算梯度
        # 根据梯度值更新参数值
        if iteration % print_every_k_iterations == 0:
            end = time.time()
            print("Iteration: {}".format(iteration),
                  "Avg. Training loss: {:.4f}".format(loss / print_every_k_iterations),
                  "{:.4f} sec/batch".format((end - start) / print_every_k_iterations))
            start = time.time()



    print(EGES.get_embedding(side_info[:, 0]))
