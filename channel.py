import math
import tensorflow as tf
import numpy as np
from sklearn.linear_model import Lasso


def channel_selection(inputs, layer, sparsity, method="lasso", data_format="channels_last"):
    """
    입력된 레이어의 출력 채널 중 프루닝 할 채널을 선택합니다.
    이 함수를 통해 현재 레이어에서 프루닝 할 채널과 해당 채널을 만드는 필터를 선택한 후 해당 채널의 인덱스를 리턴합니다.
    채널 선택을 위해, 다음 레이어의 input (X) 과 다음 레이어의 output (Y) 을 비교하게 됩니다.
    :param sparsity: float, 0 ~ 1 how many prune channel of output of this layer
    :param inputs: Tensor, input features map for next layer (corresponding to output of this layer)
    :param layer: module of the next layer (conv)
    :param method: str, how to select the channel
    :return:
        list of int, indices of filters to be pruned
    """
    num_channel = inputs.shape[-1] if data_format == "channels_last" else inputs.shape[1]   # 채널 수, next_inputs -> NHWC
    num_pruned = int(math.floor(num_channel * sparsity))    # 입력된 sparsity 에 맞춰 삭제되어야 하는 채널 수

    # lasso 방식의 channel selection
    if method == "lasso":
        if layer.use_bias:
            bias = layer.get_weights()[1].reshape((1, 1, 1, -1))
        else:
            bias = np.zeros((1, 1, 1, -1))

        outputs = layer(inputs).numpy()
        outputs = outputs - bias
        y = np.reshape(outputs, -1)
        x = []
        for i in range(num_channel):
            inputs_channel_i = np.zeros_like(inputs)
            inputs_channel_i[:, :, :, i] = inputs[:, :, :, i]
            outputs_channel_i = layer(inputs_channel_i).numpy()
            outputs_channel_i = outputs_channel_i - bias
            x.append(np.reshape(outputs_channel_i, -1))
        x = np.stack(x, axis=1)

        x = x[np.nonzero(y)]
        y = y[np.nonzero(y)]

        alpha = 1e-7
        solver = Lasso(alpha=alpha, warm_start=True, selection='random', random_state=0)

        # 원하는 수의 채널이 삭제될 때까지 alpha 값을 조금씩 늘려나감
        alpha_l, alpha_r = 0, alpha
        num_pruned_try = 0
        while num_pruned_try < num_pruned:
            alpha_r *= 2
            solver.alpha = alpha_r
            solver.fit(x, y)
            num_pruned_try = sum(solver.coef_ == 0)

        # 충분하게 pruning 되는 alpha 를 찾으면, 이후 alpha 값의 좌우를 좁혀 나가면서 좀 더 정확한 alpha 값을 찾음
        num_pruned_max = int(num_pruned * 1.1)
        while True:
            alpha = (alpha_l + alpha_r) / 2
            solver.alpha = alpha
            solver.fit(x, y)
            num_pruned_try = sum(solver.coef_ == 0)
            if num_pruned_try > num_pruned_max:
                alpha_r = alpha
            elif num_pruned_try < num_pruned:
                alpha_l = alpha
            else:
                break

        # 마지막으로, lasso coeff를 index로 변환
        indices_stayed = np.where(solver.coef_ != 0)[0].tolist()
        indices_pruned = np.where(solver.coef_ == 0)[0].tolist()

    # greedy 방식의 channel selection
    elif method == "greedy":
        channels_norm = []
        for i in range(num_channel):
            inputs_channel_i = np.zeros_like(inputs)
            inputs_channel_i[:, :, :, i] = inputs[:, :, :, i]
            outputs_channel_i = layer(inputs_channel_i).numpy()
            outputs_channel_i_norm = np.linalg.norm(outputs_channel_i)
            channels_norm.append(outputs_channel_i_norm)

        indices_pruned = np.argsort(channels_norm)[:num_pruned]

        mask = np.ones(num_channel, np.bool)
        mask[indices_pruned] = 0
        indices_stayed = np.arange(num_channel)[mask].tolist()
    else:
        raise NotImplementedError

    return indices_pruned, indices_stayed  # 선택된 채널의 인덱스를 리턴


def layer_redesign(layer, indices_stayed, design='filters'):
    """
    선택된 less important 필터/채널을 프루닝합니다.
    :param layer: layer being filters pruned
    :param indices_stayed: list of int, indices of filters/channels to be preserved
    :param design: str, 'filters' or 'channels' will be pruned
    :return:
        void
    """
    # redesign layer weight (delete filters)
    if design == 'filters':
        if isinstance(layer, tf.keras.layers.Conv2D):
            num_filters = len(indices_stayed)
            num_channels = layer.kernel.shape[2]     # W, H, in, out
            new_weight = layer.kernel.numpy()[:, :, :, indices_stayed]
            if layer.use_bias:
                new_bias = layer.bias.numpy()[indices_stayed]
            new_layer = tf.keras.layers.Conv2D(num_filters, layer.kernel_size, strides=layer.strides, padding=layer.padding, use_bias=layer.use_bias,
                                               name=layer.name)
            new_layer.build(input_shape=(None, None, None, num_channels))
            if layer.use_bias:
                new_layer.set_weights([new_weight, new_bias])
            else:
                new_layer.set_weights([new_weight])

        elif isinstance(layer, tf.keras.layers.BatchNormalization) or isinstance(layer, tf.keras.layers.Layer):
            new_gamma, new_beta = layer.weights[0].numpy()[indices_stayed], layer.weights[1].numpy()[indices_stayed]
            new_mean, new_var = layer.weights[2].numpy()[indices_stayed], layer.weights[3].numpy()[indices_stayed]
            new_layer = tf.keras.layers.BatchNormalization(axis=layer.axis, name=layer.name)
            new_layer.build(input_shape=(None, None, None, len(indices_stayed)))
            new_layer.set_weights([new_gamma, new_beta, new_mean, new_var])

    # redesign layer weight (delete channels)
    elif design == 'channels':
        if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            new_weight = layer.depthwise_kernel.numpy()[:, :, indices_stayed, :]
            if layer.use_bias:
                new_bias = layer.bias.numpy()[indices_stayed]
            new_layer = tf.keras.layers.DepthwiseConv2D(layer.kernel_size, strides=layer.strides, padding=layer.padding, use_bias=layer.use_bias,
                                                        name=layer.name)
            new_layer.build(input_shape=(None, None, None, len(indices_stayed)))
            if new_layer.use_bias:
                new_layer.set_weights([new_weight, new_bias])
            else:
                new_layer.set_weights([new_weight])

        elif isinstance(layer, tf.keras.layers.Conv2D):
            num_filters = layer.filters
            new_weight = layer.kernel.numpy()[:, :, indices_stayed, :]
            if layer.use_bias:
                new_bias = layer.bias.numpy()
            new_layer = tf.keras.layers.Conv2D(num_filters, layer.kernel_size, strides=layer.strides, padding=layer.padding, use_bias=layer.use_bias,
                                               name=layer.name)
            new_layer.build(input_shape=(None, None, None, len(indices_stayed)))
            if new_layer.use_bias:
                new_layer.set_weights([new_weight, new_bias])
            else:
                new_layer.set_weights([new_weight])

        elif isinstance(layer, tf.keras.layers.BatchNormalization) or isinstance(layer, tf.keras.layers.Layer):
            new_gamma, new_beta = layer.weights[0].numpy()[indices_stayed], layer.weights[1].numpy()[indices_stayed]
            new_mean, new_var = layer.weights[2].numpy()[indices_stayed], layer.weights[3].numpy()[indices_stayed]
            new_layer = tf.keras.layers.BatchNormalization(axis=layer.axis, name=layer.name)
            new_layer.build(input_shape=(None, None, None, len(indices_stayed)))
            new_layer.set_weights([new_gamma, new_beta, new_mean, new_var])

    return new_layer
