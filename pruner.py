import tensorflow as tf
from channel import channel_selection, layer_redesign


class MobileNetPruner():
    def __init__(self, model, train_dataset):
        self.model = model
        self.train_dataset = train_dataset

    def prune_single_block(self, layer_name, sparsity, method='greedy'):
        """
        prune filters of ith_layer
        :param layer_name: str, layer that will be filters pruned
        :param sparsity: float, amount of deleted filters
        :return:
        """
        raw_inputs, _ = next(iter(self.train_dataset))

        block_num = int(layer_name.split('_')[2])

        pw = self.model.get_layer(layer_name)
        sub = self.model.get_layer(layer_name + "_bn")

        next_dw = self.model.get_layer("conv_dw_{}".format(block_num + 1))
        next_dw_sub = self.model.get_layer("conv_dw_{}_bn".format(block_num + 1))
        next_pw = self.model.get_layer("conv_pw_{}".format(block_num + 1))

        this_inputs = raw_inputs
        for layer in self.model.layers:
            if layer is next_pw:
                break
            this_inputs = layer(this_inputs)

        indices_pruned, indices_stayed = channel_selection(this_inputs, next_pw, sparsity, method=method)

        for i in range(len(self.model._layers)):
            if self.model._layers[i] is pw:
                new_layer = layer_redesign(self.model._layers[i], indices_stayed, design='filters')
                self.model._layers[i] = new_layer
            elif self.model._layers[i] is sub:
                new_layer = layer_redesign(self.model._layers[i], indices_stayed, design='filters')
                self.model._layers[i] = new_layer
            elif self.model._layers[i] is next_dw:
                new_layer = layer_redesign(self.model._layers[i], indices_stayed, design='channels')
                self.model._layers[i] = new_layer
            elif self.model._layers[i] is next_dw_sub:
                new_layer = layer_redesign(self.model._layers[i], indices_stayed, design='channels')
                self.model._layers[i] = new_layer
            elif self.model._layers[i] is next_pw:
                new_layer = layer_redesign(self.model._layers[i], indices_stayed, design='channels')
                self.model._layers[i] = new_layer

