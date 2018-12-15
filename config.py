BATCH_SIZE = 1000
EPOCH_SIZE = 20
MAX_SEQ_LEN = 45
WORD_DIM = 100
WEIGHT_PATH='weight.h5'
MODEL_PATH ='./mymodel.mod'
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