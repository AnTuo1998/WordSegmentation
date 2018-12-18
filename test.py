import re
import sys
import json
import math
from config import MAX_SEQ_LEN,VOCAB_PATH,WEIGHT_PATH,MODEL_PATH,TEST_DATA_PATH
import numpy as np
import keras.models
from keras.preprocessing.sequence import pad_sequences
symbols = re.compile(r'，|、|《|》|（|）|？|；|-')

def division(line) -> list:
    if len(line) < MAX_SEQ_LEN  and '-' not in line:
        return [line]
    pos = symbols.search(line)
    if pos == None:
        pos = math.floor(len(line)/2)
        return division(line[0:pos]) + division(line[pos:])
    else:
        pos = pos.span()[0]
        return division(line[:pos]) + [line[pos]] + division(line[pos+1:])


def mark_splite(s,indexs,tag='  '):
    marked = ''
    for z in zip(s,indexs):
        if z[1] < 0.5:
            marked += z[0] + tag
        else:
            marked += z[0]
    return marked
    

def test(testDataDir, model):
    testData = open(testDataDir, 'r', encoding='utf-8')
    dict_file = open(VOCAB_PATH,'r', encoding='utf-8')
    vocab_dict = json.load(dict_file)
    # inverse_dict = {value:key for key, value in vocab_dict.items()}
    dict_file.close()
    res = open("result.txt", 'w', encoding='utf-8')
    count = 0
    for line in testData:
        line = division(line[:-1])
        #print(line)
        outstr = []
        for ele in line:
            if len(ele) <= 1:
                outstr.append(ele)
            else:
               # print(type(ele))
                x = np.array([vocab_dict.get(c,0) for c in ele])
                x = x.reshape(1 ,-1)    
                # print(x.shape) 
                x = pad_sequences(x, maxlen=MAX_SEQ_LEN, padding="post", truncating='post')
                y = model.predict(x).reshape(1,-1)
                # print(y.shape,y)
                predicted_y = np.array(y[0][0:len(ele)]).reshape(1,-1)[0]
                #print('predicted_y:',[float('%.1f'% i) for i in predicted_y])
                #print("predict_y size",predicted_y.shape)
                # predicted_y = predicted_y.astype(np.int)
                outstr.append(mark_splite(ele,predicted_y))
        if count <= 5:
            print(' '.join(outstr))
            count += 1
        res.write(' '.join(outstr)+"\n")
    res.close()

def test2(testDataDir, model):
    testData = open(testDataDir, 'r', encoding='utf-8')
    dict_file = open(VOCAB_PATH,'r', encoding='utf-8')
    vocab_dict = json.load(dict_file)
    # inverse_dict = {value:key for key, value in vocab_dict.items()}
    dict_file.close()
    res = open("result.txt", 'w', encoding='utf-8')
    count = 0
    for line in testData:
        line = division(line[:-1])
        #print(line)
        outstr = []
        for ele in line:
            if len(ele) <= 1:
                outstr.append(ele)
            else:
               # print(type(ele))
                x = np.array([vocab_dict.get(c,0) for c in ele])
                x = x.reshape(1 ,-1)    
                # print(x.shape) 
                x = pad_sequences(x, maxlen=MAX_SEQ_LEN, padding="post", truncating='post')
                y = model.predict(x).reshape(1,-1)
                # print(y.shape,y)
                predicted_y = np.array(y[0][0:len(ele)]).reshape(1,-1)[0]
                #print('predicted_y:',[float('%.1f'% i) for i in predicted_y])
                #print("predict_y size",predicted_y.shape)
                # predicted_y = predicted_y.astype(np.int)
                outstr.append(mark_splite(ele,predicted_y))
        if count <= 5:
            print('  '.join(outstr))
            count += 1
        res.write('  '.join(outstr)+"\n")
    res.close()


if __name__ == '__main__':
    model = keras.models.load_model(MODEL_PATH+sys.argv[1]+'.mod')
    model.load_weights(WEIGHT_PATH+sys.argv[1]+'.h5')
    test2(TEST_DATA_PATH,model)
    