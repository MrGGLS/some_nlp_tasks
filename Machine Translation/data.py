import torch
from datasets import load_dataset, load_from_disk

dataset = load_from_disk('wmt_zh-en.hf')

special_tokens = ['<START>', '<PAD>', '<END>']

def get_dict(dataset):
  lens = len(dataset)
  src, ref = '', ''
  vocab_dic = {}
  for data in dataset:
    src += data['source']
    ref += data['reference'] + ' '
  vocab = special_tokens + list(src) + ref.split(' ') 
  vocab = set(vocab)
  vocab_dic['encode'] = {word: idx for idx, word in enumerate(vocab)}
  vocab_dic['decode'] = {idx: word for idx, word in enumerate(vocab)}
  return vocab_dic

vocab_dic = get_dict(dataset)

def encode_seq(dic, seq):
  encoded = [dic['encode'][word] for word in seq]
  return encoded

def decode_seq(dic, seq):
  decoded = [dic['decode'][idx] for idx in seq]
  return decoded

class Dataset(torch.utils.data.Dataset):
  def __init__(self, dataset):
    def f(data):
      if len(data['source']) <= 78 and len(data['reference']) <= 79:
        return data
    self.dataset = dataset.filter()

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, i):
    src = ['<START>'] + list(self.dataset[i]['source']) + ['<END>']
    ref = ['<START>'] + self.dataset[i]['reference'].split(' ') + ['<END>']
    src = src + ['<PAD>'] * 80
    ref = ref + ['<PAD>'] * 81
    src = src[:80]
    ref = ref[:81]
    return torch.LongTensor(encode_seq(vocab_dic, src)), torch.LongTensor(encode_seq(vocab_dic, ref))

loader = torch.utils.data.DataLoader(dataset=Dataset(dataset),
                                     batch_size=16,
                                     drop_last=True,
                                     shuffle=True,
                                     collate_fn=None)

