from collections import Counter
import config
import json
import re
import math
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
import numpy as np

def infer_label(word_list) -> list:
    """0 = not 1 = connect"""
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
    
def getVocabDict(X) -> dict:
    """build a vocabulary dictionary containing every words"""
    from config import VOCAB_PATH
    try:
        dict_file = open(VOCAB_PATH, 'r', encoding='utf-8')
        voc_dic = json.load(dict_file)
        dict_file.close()
    except (NameError,OSError,FileNotFoundError):
        voc = set()
        for x in X:
            voc.update(x)
        voc_dic = {k:i+1 for i,k in enumerate(voc)}
        voc_dic['unknown-word'] = 0
        dict_file = open(VOCAB_PATH, 'w', encoding='utf-8')
        json.dump(voc_dic,dict_file)
        dict_file.close()
    finally:
        print('vocabulary length:',len(voc_dic))
        return voc_dic

def selectSeqLen(X) -> int:
    """Get a suitable len of a sentence"""
    x_lengths = [len(x) for x in X]
    # plt.hist(x_lengths,bins=100,stacked=True,)#rwidth=0.8
    plt.hist(x_lengths)
    plt.savefig(config.FIGURE_PATH + 'data.png')
    
    print(max(x_lengths))
    len_counts = Counter(x_lengths)

    top_n = len_counts.most_common(30)# 出现频率最高的序列长度
    print(top_n)
    seqLen =  max(top_n,key=lambda t:t[0])[0]
    #if seqLen < 45:
     #   seqLen = 45
    print('selected MAX_SEQ_LENGTH：',seqLen)
    s_num = 0
    for k,v in len_counts.items():
        if k <= seqLen: 
            s_num+=v
    print('长度小于{}的句子数:{}。共有句子{}'.format(seqLen, s_num, len(x_lengths)))
    return seqLen

def division(line :str) -> list:
    # flag = True
    from config import MAX_SEQ_LEN
    if len(line) < MAX_SEQ_LEN:
        return [line]
    sentenceList = []
    SplitWordStr = "，。！？、／：；《》（）、"
    sentence = ""

    for i in range(len(line)):
        sentence += line[i]
        if line[i] in SplitWordStr and i + 1 < len(line) and line[i + 1] not in SplitWordStr:
            sentenceList.append(sentence)
            sentence = ""

    if sentence != "":
        sentenceList.append(sentence)
    return sentenceList

def processTrainData(trainDataDir, SeqLen=True) -> (list,list):
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
    
    X = np.array([[vocabDict[c] for c in x] for x in X])
    config.setWordNum(len(vocabDict))
    print(len(vocabDict))
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

def processTrainData2(trainDataDir, SeqLen=True) -> (list,list):
    "get data and decide length of sentence to train"
    X = []
    y = []
    train_data = open(trainDataDir,'r',encoding='utf-8')
    count = 0
    for line in train_data:
        sentenceList = division(line[0:-1])
        for sentence in sentenceList:
            if len(sentence) <= 1:
                continue
            one_list = sentence.split()
            label = infer_label(one_list)
            X.append(list(''.join(one_list)))
            y.append(label)
        count += 1
        if count % 10 == 0:
            print("process {:3f}%\r".format(count * 100 / 86924), end='')
    train_data.close()
    X = np.array(X)
    y = np.array(y)
    print(X[0])
    print(y[0])

    vocabDict = getVocabDict(X)
    
    X = np.array([[vocabDict[c] for c in x] for x in X])
    config.setWordNum(len(vocabDict))
    print(len(vocabDict))
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


