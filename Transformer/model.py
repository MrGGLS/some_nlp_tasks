import torch.nn
from utils import MultiHeads, FeedForwardNet, PositionEmbedding
from mask import mask_pad, mask_tril


class EncoderLayer(torch.nn.Module):
    def __init__(self, heads, embed_dims, drop_rate):
        super(EncoderLayer, self).__init__()
        self.mhs = MultiHeads(heads=heads, embed_dims=embed_dims, drop_rate=drop_rate)
        self.ff = FeedForwardNet(embeds_dim=embed_dims, drop_rate=drop_rate)

    def forward(self, x, mask):
        score = self.mhs(x, x, x, mask)
        return self.ff(score)


class Encoder(torch.nn.Module):
    def __init__(self, heads, embed_dims, drop_rate):
        super(Encoder, self).__init__()
        self.l1 = EncoderLayer(heads=heads, embed_dims=embed_dims, drop_rate=drop_rate)
        self.l2 = EncoderLayer(heads=heads, embed_dims=embed_dims, drop_rate=drop_rate)
        self.l3 = EncoderLayer(heads=heads, embed_dims=embed_dims, drop_rate=drop_rate)

    def forward(self, x, mask):
        x = self.l1(x, mask)
        x = self.l2(x, mask)
        return self.l3(x, mask)


class DecoderLayer(torch.nn.Module):
    def __init__(self, heads, embed_dims, drop_rate):
        super(DecoderLayer, self).__init__()
        self.mhs1 = MultiHeads(heads=heads, embed_dims=embed_dims, drop_rate=drop_rate)
        self.mhs2 = MultiHeads(heads=heads, embed_dims=embed_dims, drop_rate=drop_rate)
        self.ff = FeedForwardNet(embeds_dim=embed_dims, drop_rate=drop_rate)

    def forward(self, x, y, mask_pad_x, mask_tril_y):
        y_self_attens = self.mhs1(y, y, y, mask_tril_y)
        y_attens = self.mhs2(y_self_attens, x, x, mask_pad_x)
        return self.ff(y_attens)


class Decoder(torch.nn.Module):
    def __init__(self, heads, embed_dims, drop_rate):
        super(Decoder, self).__init__()
        self.l1 = DecoderLayer(heads=heads, embed_dims=embed_dims, drop_rate=drop_rate)
        self.l2 = DecoderLayer(heads=heads, embed_dims=embed_dims, drop_rate=drop_rate)
        self.l3 = DecoderLayer(heads=heads, embed_dims=embed_dims, drop_rate=drop_rate)

    def forward(self, x, y, mask_pad_x, mask_tril_y):
        y = self.l1(x, y, mask_pad_x, mask_tril_y)
        y = self.l1(x, y, mask_pad_x, mask_tril_y)
        return self.l1(x, y, mask_pad_x, mask_tril_y)


class Transformer(torch.nn.Module):
    def __init__(self, words_size, sens_len, heads, embed_dims, drop_rate):
        super(Transformer, self).__init__()
        self.embed_x = PositionEmbedding(words_size=words_size, sens_len=sens_len, embeds_dim=embed_dims)
        self.embed_y = PositionEmbedding(words_size=words_size, sens_len=sens_len, embeds_dim=embed_dims)
        self.encoder = Encoder(heads=heads, embed_dims=embed_dims, drop_rate=drop_rate)
        self.decoder = Decoder(heads=heads, embed_dims=embed_dims, drop_rate=drop_rate)
        self.fc_out = torch.nn.Linear(embed_dims, words_size)

    def forward(self, x, y):
        mask_pad_x = mask_pad(x)
        mask_tril_y = mask_tril(y)

        x, y = self.embed_x(x), self.embed_y(y)
        x = self.encoder(x, mask_pad_x)
        y = self.decoder(x, y, mask_pad_x, mask_tril_y)
        return self.fc_out(y)
