import time
import argparse
import tensorflow as tf
from data_pipe import *


def test():

    parser = argparse.ArgumentParser()
    parser.add_argument('--original', type=str, default=None, help='str, original model file name (h5 format)')
    parser.add_argument('--pruned', type=str, default=None, help='str, pruned model file name (h5 format)')
    args = parser.parse_args()

    if isinstance(args.original, str):
        original_model = tf.keras.models.load_model(args.original)
    else:
        original_model = tf.keras.applications.MobileNet(weights='imagenet')

    if isinstance(args.pruned, str):
        pruned_model = tf.keras.models.load_model(args.pruned)
    else:
        args.pruned = "mobilenet_0.h5"
        pruned_model = tf.keras.models.load_model(args.pruned)

    ori_time = tf.keras.metrics.Mean()
    pru_time = tf.keras.metrics.Mean()
    ori_acc = tf.keras.metrics.Mean()
    pru_acc = tf.keras.metrics.Mean()

    data_dir = "imagenet"
    for i in range(10):
        part_label = int(args.pruned.split('_')[-1].split('.')[0])
        train_dataset, test_dataset = load_imagenet_tf_dataset_part(data_dir, batch_size=64,
                                                                    part_labels=part_label)
        ori_n, ori_cor = 0, 0
        for x, y in test_dataset:
            start = time.time()
            pred = original_model(x)
            end = time.time() - start
            ori_n += x.shape[0]
            pred = np.argmax(pred, axis=1)
            pred_part = np.equal(pred, part_label)
            y = np.argmax(y, axis=1)
            y_part = np.equal(y, part_label)
            ori_cor += np.sum(np.equal(pred_part, y_part))
        ori_time(end)
        ori_acc(ori_cor / ori_n)
        print("original_acc: {}".format(ori_cor / ori_n))

        pru_n, pru_cor = 0, 0
        for x, y in test_dataset:
            start = time.time()
            pred = pruned_model(x)
            end = time.time() - start
            pru_n += x.shape[0]
            pred = np.argmax(pred, axis=1)
            pred_part = np.equal(pred, part_label)
            y = np.argmax(y, axis=1)
            y_part = np.equal(y, part_label)
            pru_cor += np.sum(np.equal(pred_part, y_part))
        pru_time(end)
        pru_acc(pru_cor / pru_n)
        print("pruned_acc: {}".format(pru_cor / pru_n))

    print("original_time: {t}       original_acc: {a}".format(t=ori_time.result().numpy(), a=ori_acc.result().numpy()))
    print("pruned_time: {t}         pruned_acc: {a}".format(t=pru_time.result().numpy(), a=pru_acc.result().numpy()))


if __name__ == '__main__':
    test()
