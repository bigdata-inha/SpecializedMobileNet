import argparse
import re
import tensorflow as tf
from data_pipe import *
from pruner import MobileNetPruner


def extraction():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original', type=str, default=None, help='str, original model file name (h5 format)')
    parser.add_argument('--category', type=int, default=0, help='int, which category is model adapted')
    parser.add_argument('--sparsity', type=float, default=0.1, help='0.1 ~ 0.9, how many are filters deleted for each layer')
    args = parser.parse_args()

    if isinstance(args.original, str):
        original_model = tf.keras.models.load_model(args.original)
    else:
        original_model = tf.keras.applications.MobileNet(weights='imagenet')
        args.original = "mobilenet_original.h5"
        original_model.save(args.original)
    print("original model parameter:", original_model.count_params())

    data_dir = "imagenet"
    part_label = args.category
    train_dataset, test_dataset = load_imagenet_tf_dataset_part(data_dir, batch_size=64, part_labels=part_label)

    sparsity = args.sparsity
    model = tf.keras.models.load_model(args.original)

    print("pruning start")
    pruner = MobileNetPruner(model, train_dataset)

    for layer in reversed(model.layers):
        if bool(re.fullmatch('conv_pw_\d*', layer.name)):
            if layer.name in 'conv_pw_13':
                continue
            pruner.prune_single_block(layer.name, sparsity, method='greedy')

    print("pruning end")
    pruned_model = tf.keras.Sequential()
    for layer in pruner.model.layers:
        pruned_model.add(layer)

    model.save("mobilenet_" + str(part_label) + ".h5")

    print("pruned model parameter:", pruned_model.count_params())


if __name__ == '__main__':
    extraction()
