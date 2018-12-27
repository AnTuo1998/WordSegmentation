import re
import sys
import json
import math
from config import MAX_SEQ_LEN,VOCAB_PATH,MODEL_PATH,TEST_DATA_PATH
import numpy as np
import keras.models
from keras.preprocessing.sequence import pad_sequences

symbols = re.compile(r'，|、|《|》|（|）|？|；|-')

def division(line: str) -> list:
    from config import MAX_SEQ_LEN
    if len(line) < MAX_SEQ_LEN:
        return [line]
    pos = symbols.search(line)
    if pos == None:
        pos = math.floor(len(line)/2)
        return division(line[0:pos]) + division(line[pos:])
    else:
        pos = pos.span()[0]
        return division(line[:pos])+ division(line[pos]) + division(line[pos+1:])

def merge(s,indexs,tag='  '):
    marked = ''
    for z in zip(s,indexs):
        if z[1] < 0.5:
            marked += z[0] + tag
        else:
            marked += z[0]
    return marked

def test(testDataDir:str, model) -> None:
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
                x = np.array([vocab_dict.get(c,0) for c in ele])
                x = x.reshape(1 ,-1)    
                x = pad_sequences(x, maxlen=MAX_SEQ_LEN, padding="post", truncating='post')
                y = model.predict(x).reshape(1,-1)
                predicted_y = np.array(y[0][0:len(ele)]).reshape(1,-1)[0]
                outstr.append(merge(ele,predicted_y))
        if count < 5:
            print('  '.join(outstr))
        count += 1
        if count % 5 == 0:
            print('\rprocess {:3f}%'.format(count / 3985 * 100),end='')
        res.write('  '.join(outstr)+"\n")
    testData.close()
    res.close()


if __name__ == '__main__':
    try:
        model = keras.models.load_model(MODEL_PATH+sys.argv[1]+'.hdf5')
        model.summary()
        test(TEST_DATA_PATH,model)
    except OSError:
        print("Can't Find Model named "+sys.argv[1])