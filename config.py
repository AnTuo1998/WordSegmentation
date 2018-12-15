BATCH_SIZE = 5000
EPOCH_SIZE = 20
MAX_SEQ_LEN = 45
WORD_DIM = 100
WORD_NUM = 0
def setSeqLen(len):
    global MAX_SEQ_LEN
    MAX_SEQ_LEN = len
    print(MAX_SEQ_LEN)

def getSeqLen():
    return MAX_SEQ_LEN

def setWordNum(num):
    global WORD_NUM
    WORD_NUM = num

TRAIN_DATA_PATH = './train.txt'
WEIGHT_PATH='./model/weight.h5'
MODEL_PATH ='./model/mymodel.mod'
FIGURE_PATH = './fig/'
VOCAB_PATH = './data/vocabulary.json'