import random
import numpy as np
import torch
from torch.utils.data import Dataset

vocabs = '<START>,<PAD>,<END>,<GGLS>,' \
         '0,1,2,3,4,5,6,7,8,9,' \
         'q,w,e,r,t,y,u,i,o,p,a,s,d,f,g,h,j,k,l,z,x,c,v,b,n,m,' \
         'Q,W,E,R,T,Y,U,I,O,P,A,S,D,F,G,H,J,K,L,Z,X,C,V,B,N,M'.split(',')

v2i = {word: idx for idx, word in enumerate(vocabs)}
i2v = {idx: word for idx, word in enumerate(vocabs)}


def get_data():
    words = vocabs[3:]
    n = random.randint(30, 48)
    x = np.random.choice(words, size=n, replace=True).tolist()

    def generate_y():
        for i, c in enumerate(x):
            if not c.isdigit():
                x[i] = c.upper()
            else:
                x[i] = str(9 - int(x[i]))
        y = x + [x[-1]]
        return y[::-1]

    y = generate_y()

    x = ['<START>'] + x + ['<END>']
    y = ['<START>'] + ['<GGLS>'] +y + ['<END>']
    x = x + ['<PAD>'] * 50
    x = x[:50]
    y = y + ['<PAD>'] * 51
    y = y[:51]
    x = [v2i[w] for w in x]
    y = [v2i[w] for w in y]

    x = torch.LongTensor(x)
    y = torch.LongTensor(y)

    return x, y


class CustomDataset(Dataset):
    def __init__(self):
        super(Dataset, self).__init__()

    def __len__(self):
        return 100000

    def __getitem__(self, item):
        return get_data()


loader = torch.utils.data.DataLoader(dataset=CustomDataset(),
                                     batch_size=16,
                                     drop_last=True,
                                     shuffle=True,
                                     collate_fn=None)