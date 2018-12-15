import re
import json
from config import MAX_SEQ_LEN,VOCAB_PATH
import numpy as np
from keras.preprocessing.sequence import pad_sequences
symbols = re.compile(r'，|。|《|》|（|）|？|。|；')

def division(line) -> list:
    pos = symbols.search(line)
    if pos == None:
        if len(line) < MAX_SEQ_LEN:
            return [line]
        else:
            return division(line[0: len(line/2)]) + division(line[len(line)/2:])
    else:
        pos = pos.span()[0]
        return [line[:pos], line[pos]] + division(line[pos+1:])


def mark_splite(s,indexs,tag='  '):
    for z in zip(s,indexs):
        if z[1] < 0.5:
            marked += z[0] + tag
        else:
            marked += z[0]
    return marked
    

def getTestData(testDataDir, model):
    testData = open(testDataDir, 'r', encoding='utf-8')

    dict_file = open(VOCAB_PATH,'r', encoding='utf-8')
    vocab_dict = json.load(dict_file)
    dict_file.close()
    res = open("result.txt", 'w', encoding='utf-8')
    count = 0
    for line in testData:
        line = division(line)
        outstr = []
        for ele in line:
            if len(ele) <= 1:
                outstr.append(ele)
            else:
                print(type(ele))
                x = np.array([vocab_dict.get(c,0) for c in ele])
                x = x.reshape(1 ,-1)    
                print(x.shape) 
                x = pad_sequences(x, maxlen=MAX_SEQ_LEN, padding="post", truncating='post');
                y = model.predict(x).reshape(1,-1)
                # print(y.shape,y)
                predicted_y = np.array(y[0][0:len(ele)]).reshape(1,-1)[0]
                #print('predicted_y:',[float('%.1f'% i) for i in predicted_y])
                #print("predict_y size",predicted_y.shape)
                # predicted_y = predicted_y.astype(np.int)
                outstr.append(mark_splite(ele,predicted_y))
        if count >= 5:
            print(' '.join(outstr))
            count+=1
        res.write(' '.join(outstr)+'\n')
    res.close()
