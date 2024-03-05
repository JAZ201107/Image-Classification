import torch


def loss_fn(outputs, labels):

    num_examples = outputs.size()[0]

    return -torch.sum(outputs[range(num_examples), labels]) / num_examples
