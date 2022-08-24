# Some Exercises For NLP Tasks

## Build  A Transformer from Scratch

Build a easy-transformer and do some things fun

### Task Target

Given an input sequence with letters and numbers. We will gonna do is to reverse this sequence and add a special token`<GGLS>` in the begin of the new sequence（`<xxx>` means which is a special token）

```shell
# Needn't care about these special tokens, just focus on sequences
# input
x: <START>ODM5KYWBSGBRNLFTUS8DUJEHVD6SNFFHWD2VF6NY<END><PAD><PAD><PAD><PAD><PAD><PAD><PAD><PAD>
# target
y: <START><GGLS>YYN6FV2DWHFFNS6DVHEJUD8SUTFLNRBGSBWYK5MDO<END><PAD><PAD><PAD><PAD><PAD><PAD><PAD>
# output
pred: <START><GGLS>YYN6FV2DWHFFNS6DVHEJUD8SUTFLNRBGSBWYK5MDO<END>D<END>M<END><END><END>
```

## Advanced Target

Let input x not only contains uppercases but also lowercases, but target's all letters should be all upper

```
# input
x: <START>aBcD123<END><PAD><PAD><PAD><PAD><PAD><PAD><PAD><PAD>
# target
y: <START><GGLS>321DCBA<END><PAD><PAD><PAD><PAD><PAD><PAD><PAD>
# output
pred: <START><GGLS>321DCBA<END>D<END>M<END><END><END>
```

Or maybe somthing harder?

## Pretrained Model From HuggingFace

Use pretrained models from huggingface to do some Chinese NLP tasks

### Sentiment Analysis

Is sentence positive or negative?

### Fill In The Blank

Which word should be fill in the blank?

### Relation Between Two Parts

Are these two sequences from one sentence?

### Named Entity Recognition

Where are these named entities in this ssentence?

## End

Our transformer can work well without GPU but those pretrained models are not...(extremely slow)

You could see what I did [here](https://colab.research.google.com/drive/1jaY8J6lWOmIlZYel7ADKxvR5gFgu_aWd?usp=sharing)