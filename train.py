import matplotlib.pyplot as plt
import numpy as np
import sys
import processData
from sklearn.model_selection import train_test_split
from buildModel import build_model
from keras.callbacks import ModelCheckpoint,TensorBoard
from test import test2
from config import TRAIN_DATA_PATH,EPOCH_SIZE,BATCH_SIZE,\
                WEIGHT_PATH,MODEL_PATH,FIGURE_PATH,TEST_DATA_PATH


def myplot(history):
    """plot loss and acc"""
    epoch = range(1, EPOCH_SIZE + 1)
    _, axes = plt.subplots(nrows = 2, ncols = 1, figsize = (12, 12))
    axes[0].plot(epoch, history.history['acc'], label = 'train')
    axes[0].plot(epoch, history.history['val_acc'], label = 'validation')
    axes[0].legend()
    axes[0].grid(axis = 'y')
    axes[0].set_xlabel(xlabel = 'epoch')
    axes[0].set_ylabel(ylabel = 'accuracy')
    axes[0].set_title(label = 'Accuracy')
    axes[1].plot(epoch, history.history['loss'], label = 'train')
    axes[1].plot(epoch, history.history['val_loss'], label = 'validation')
    axes[1].legend()
    axes[1].grid(axis = 'y')
    axes[1].set_xlabel(xlabel = 'epoch')
    axes[1].set_ylabel(ylabel = 'loss')
    axes[1].set_title(label = 'Loss')
    plt.savefig(FIGURE_PATH+'analysis.png')

def train(name=None):
    model,isTrain = build_model(name)
    if isTrain:
        return model

    X, y = processData.processTrainData(TRAIN_DATA_PATH)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

    checkpointer = ModelCheckpoint(filepath=WEIGHT_PATH+name+'-{epoch:02d}e-val_acc_{val_acc:.2f}.hdf5', \
                                verbose=1,save_best_only=True)
    print("begin training")
    print(EPOCH_SIZE,BATCH_SIZE)
   #  from config import EPOCH_SIZE,BATCH_SIZE
    history = model.fit(X_train,y_train,validation_data=(X_val,y_val), 
                    epochs=EPOCH_SIZE, batch_size=BATCH_SIZE, callbacks=[checkpointer,TensorBoard(log_dir='./logs/log')])
    model.save(MODEL_PATH+name+'.mod')
    model.save_weights(WEIGHT_PATH+name+'.h5')

    myplot(history)
    return model


if __name__ == "__main__":
    model = None
    if len(sys.argv) > 1:
        model = train(sys.argv[1])
    else:
        model = train()
    
    test2(TEST_DATA_PATH, model)

