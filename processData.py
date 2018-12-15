from collections import Counter
import config
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
import numpy as np


def infer_label(word_list) -> list:
    "0 = not 1 = connect"
    label = []
    for word in word_list:
        if len(word) == 1:
            label.append(0)
        else:
            count = len(word)
            while count > 1:
                count -= 1
                label.append(1)
            label.append(0)
    return label
    

def getVocabDict(X):
    "build a vocabulary dictionary containing every words"
    voc = set()
    for x in X:
        voc.update(x)
    voc_dic = {k:i+1 for i,k in enumerate(voc)}
    voc_dic['unknown-word'] = 0
    print('vocabulary length:',len(voc_dic))

    with open('./vocabulary_dict.json','wt',encoding='utf8') as f:
        json.dump(voc_dic,f)
    return voc_dic

def selectSeqLen(X):
    x_lengths = [len(x) for x in X]
    # plt.hist(x_lengths,bins=100,stacked=True,)#rwidth=0.8
    # plt.show()
    print(max(x_lengths))
    len_counts = Counter(x_lengths)

    top_n = len_counts.most_common(30)# 出现频率最高的序列长度
    print(top_n)
    seqLen =  max(top_n,key=lambda t:t[0])[0]
    print('selected MAX_SEQ_LENGTH：',seqLen)
    s_num = 0
    for k,v in len_counts.items():
        if k <= seqLen: 
            s_num+=v
    print('长度小于{}的句子数:{}。共有句子{}'.format(seqLen, s_num, len(x_lengths)))
    return seqLen


def processTrainData(trainDataDir, SeqLen=True) -> dict:
    "get data and decide length of sentence to train"
    X = []
    y = []
    train_data = open(trainDataDir,'r',encoding='utf-8')
    for line in train_data:
        one_list = line.split()
        label = infer_label(one_list)
        X.append(list(''.join(one_list)))
        y.append(label)    
    train_data.close()
    X = np.array(X)
    y = np.array(y)
    print(X[0])
    print(y[0])
    vocabDict = getVocabDict(X)
    from config import VOCAB_PATH
    dict_file = open(VOCAB_PATH, 'w', encoding='utf-8')
    json.dump(vocabDict,dict_file)
    dict_file.close()
    X = np.array([[vocabDict[c] for c in x] for x in X])
    config.setWordNum(len(vocabDict))

    if SeqLen:
        seqLen = selectSeqLen(X)
    else:
        seqLen = 45
    config.setSeqLen(seqLen)

    from config import MAX_SEQ_LEN
    X = pad_sequences(X, maxlen=MAX_SEQ_LEN, padding='post', truncating='post')
    y = pad_sequences(y, maxlen=MAX_SEQ_LEN, padding='post', truncating='post')
    # transform y into 3-dim tensor
    y = y.reshape(-1,MAX_SEQ_LEN,1) 
    return X, y


