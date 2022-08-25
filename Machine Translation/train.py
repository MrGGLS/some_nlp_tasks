import torch

from model import Transformer
from data import v2i, i2v, loader
from mask import mask_pad, mask_tril
from tqdm import tqdm


def predict(x):
    '''
    :param x: [1, n]
    :return: predicted sentence
    '''
    n = x.shape[1]
    mask_pad_x = mask_pad(x)
    pred = [v2i['<START>']] + [v2i['<PAD>']] * (n - 1)
    pred = torch.LongTensor(pred).unsqueeze(0)
    x = transformer.embed_x(x)
    x = transformer.encoder(x, mask_pad_x)

    for i in range(n - 1):
        y = pred
        mask_tril_y = mask_tril(y)
        y = transformer.embed_y(y)
        y = transformer.decoder(x, y, mask_pad_x, mask_tril_y)
        out = transformer.fc_out(y)
        out = out[:, i, :].argmax(dim=1).detach()
        pred[:, i + 1] = out

    return pred

    for i in range(n - 1):
        y = pred
        mask_tril_y = mask_tril(y)
        embed_y = model.embed_y(y)
        decoded_y = model.decoder(embed_x, embed_y, mask_pad_x, mask_tril_y)
        out = model.fc_out(decoded_y)
        out = out[:, i, :].argmax(dim=1).detach()
        pred[:, i + 1] = out

    return pred


lr = 1e-3
words_size = len(v2i)
embed_dims = 32
drop_rate = 0.1
heads = 4
sens_len = 50
epochs = 1

if __name__ == '__main__':
    transformer = Transformer(words_size=words_size, sens_len=sens_len, heads=heads, embed_dims=embed_dims,
                              drop_rate=drop_rate)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(transformer.parameters(), lr=lr)

    for epoch in range(epochs):
        pbar = tqdm(enumerate(loader), total=len(loader))
        for i, (x, y) in pbar:
            pred = transformer(x, y[:, :-1])
            pred = pred.reshape(-1, words_size)
            y = y[:, 1:].reshape(-1)
            selected = y != v2i['<PAD>']
            pred = pred[selected]
            y = y[selected]
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred = pred.argmax(dim=1)
            pbar.set_description(
                f"epoch {epoch + 1} iter {i}: train loss {loss.item():.5f}. lr {lr:e} acc {((pred == y).sum().item()) / len(pred)}")
    torch.save(transformer.state_dict(), 'transformer.pt')

    # see how well our transformer works
    # def toStr(data):
    #     data = data.tolist()
    #     str = ''
    #     for idx in data:
    #         str += i2v[idx]
    #     return str

    # for i, (x, y) in enumerate(loader):
    #     print('x:', toStr(x[0]))
    #     print('y:', toStr(y[0]))
    #     print('pred:', toStr(predict(x[0].unsqueeze(0))[0]))
    #     break
