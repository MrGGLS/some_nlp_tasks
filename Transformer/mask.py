import torch

from data import v2i


def mask_pad(data):
    '''
    :param data: [b, n] a batch of sentences
    :return: mask: [b, 1, n, n]
    '''
    n = data.shape[1]
    mask = data == v2i['<PAD>']
    mask = mask.reshape(-1, 1, 1, n)
    mask = mask.expand(-1, 1, n, n)
    return mask


def mask_tril(data):
    '''
    :param data: [b, n] a batch of sentences
    :return: mask: [b, 1, n, n]
    '''
    n = data.shape[1]
    tril = 1 - torch.tril(torch.ones(1, n, n, dtype=torch.long))
    mask = data == v2i['<PAD>']
    mask = mask.reshape(-1, 1, n).long()
    mask = mask + tril
    mask = (mask > 0).reshape(-1, 1, n, n)
    return mask
