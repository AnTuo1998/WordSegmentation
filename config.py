BATCH_SIZE = 256
EPOCH_SIZE = 20
MAX_SEQ_LEN = 45
WORD_DIM = 50
WORD_NUM = 5168
def setSeqLen(len):
    global MAX_SEQ_LEN
    MAX_SEQ_LEN = len
    print(MAX_SEQ_LEN)

def getSeqLen():
    return MAX_SEQ_LEN

def setWordNum(num):
    global WORD_NUM
    WORD_NUM = num

TRAIN_DATA_PATH = './data/train.txt'
TEST_DATA_PATH = './data/test.txt'
WEIGHT_PATH='./model/weight.h5'
MODEL_PATH ='./model/mymodel.mod'
FIGURE_PATH = './figure/'
VOCAB_PATH = './data/vocabulary.json'