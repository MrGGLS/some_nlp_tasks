import math

import torch


def attention(Q, K, V, mask):
    '''
    :param Q: [b, hs, n, e/hs]
    :param K: [b, hs, n, e/hs]
    :param V: [b, hs, n, e/hs]
    :param mask: [b, 1, n, n]
    :return: score: [b, n, e]

    calculate attention scores and return

    '''
    # [b, hs, n, e/hs] x [b, hs, e/hs, n] -> [b, hs, n, n]
    score = torch.matmul(Q, K.permute(0, 1, 3, 2))
    score /= Q.shape[3] ** 0.5
    score = score.masked_fill_(mask, -float('inf'))
    score = torch.softmax(score, dim=-1)
    # [b, hs, n, n] x [b, hs, n, e/hs] -> [b, hs, n, e/hs] -> [b, n, hs, e/hs] -> [b, n, e]
    score = torch.matmul(score, V).permute(0, 2, 1, 3).reshape(Q.shape[0], Q.shape[2], -1)
    return score


class MultiHeads(torch.nn.Module):
    def __init__(self, heads, embed_dims, drop_rate):
        super(MultiHeads, self).__init__()
        self.heads = heads
        self.QW = torch.nn.Linear(embed_dims, embed_dims)
        self.KW = torch.nn.Linear(embed_dims, embed_dims)
        self.VW = torch.nn.Linear(embed_dims, embed_dims)
        self.out_fc = torch.nn.Linear(embed_dims, embed_dims)
        self.norm = torch.nn.LayerNorm(normalized_shape=embed_dims, elementwise_affine=True)
        self.dropout = torch.nn.Dropout(p=drop_rate)

    def forward(self, Q, K, V, mask):
        '''
        :param Q: [b, n, e]
        :param K: [b, n, e]
        :param V: [b, n, e]
        :param mask: [b, 1, n, n]
        :return: score [b, n, e]
        '''
        cloned_Q = Q.clone()
        b = Q.shape[0]
        n = Q.shape[1]
        e = Q.shape[2]

        Q = self.norm(Q)
        K = self.norm(K)
        V = self.norm(V)

        # [b, n, e] -> [b, hs, n, e/hs]
        Q = self.QW(Q).reshape(b, n, self.heads, e // self.heads).permute(0, 2, 1, 3)
        K = self.KW(K).reshape(b, n, self.heads, e // self.heads).permute(0, 2, 1, 3)
        V = self.VW(V).reshape(b, n, self.heads, e // self.heads).permute(0, 2, 1, 3)

        score = attention(Q, K, V, mask)

        score = score + cloned_Q

        return score


class PositionEmbedding(torch.nn.Module):
    def __init__(self, words_size, sens_len, embeds_dim):
        super(PositionEmbedding, self).__init__()

        def get_pe(pos, i, d_model):
            pe = pos / (1e4 ** (i / d_model))

            if i % 2 == 0:
                return math.sin(pe)
            return math.cos(pe)

        pe = torch.empty(sens_len, embeds_dim)
        for i in range(sens_len):
            for j in range(embeds_dim):
                pe[i, j] = get_pe(i, j, embeds_dim)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

        self.embeds = torch.nn.Embedding(words_size, embeds_dim)

    def forward(self, x):
        embeds = self.embeds(x)
        embeds = embeds + self.pe
        return embeds


class FeedForwardNet(torch.nn.Module):
    def __init__(self, embeds_dim, drop_rate):
        super(FeedForwardNet, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(embeds_dim, embeds_dim * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(embeds_dim * 2, embeds_dim),
            torch.nn.Dropout(p=drop_rate)
        )
        self.norm = torch.nn.LayerNorm(normalized_shape=embeds_dim, elementwise_affine=True)

    def forward(self, x):
        cloned_x = x.clone()

        x = self.norm(x)
        out = self.fc(x)
        out = cloned_x + out

        return out
