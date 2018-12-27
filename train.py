import matplotlib.pyplot as plt
import numpy as np
import sys
import keras.models
from processData import processTrainData,processTrainData2
from sklearn.model_selection import train_test_split
from buildModel import build_model
from keras.callbacks import ModelCheckpoint,TensorBoard
from test import test
from config import TRAIN_DATA_PATH,EPOCH_SIZE,BATCH_SIZE,\
                    MODEL_PATH,FIGURE_PATH,TEST_DATA_PATH

def myplot(history, name):
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
    plt.savefig(FIGURE_PATH+name+'.analysis.png')

def train(name: str) ->keras.models:
    model,isTrain = build_model(name)
    if isTrain:
        return model

    X, y = processTrainData2(TRAIN_DATA_PATH,False)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

    checkpointer = ModelCheckpoint(filepath=MODEL_PATH+name+'{epoch:02d}e-val_acc_{val_acc:.2f}.hdf5', verbose=1, save_best_only=True)
    print("begin training")
    print(EPOCH_SIZE,BATCH_SIZE)
   #  from config import EPOCH_SIZE,BATCH_SIZE
    history = model.fit(X_train,y_train,validation_data=(X_val,y_val), verbose=1,
                    epochs=EPOCH_SIZE, batch_size=BATCH_SIZE, callbacks=[checkpointer,TensorBoard(log_dir='./logs')])
    model.save(MODEL_PATH+name+'.hdf5')

    myplot(history,name)
    return model


if __name__ == "__main__":
    if len(sys.argv) > 1:
        modelName = sys.argv[1]
        if '.hdf5' in modelName:
            modelName = model[:-5]
        if MODEL_PATH in modelName:
            modelName = modelName[8:]
        model = train(sys.argv[1])
        test(TEST_DATA_PATH, model)
    else:
        print("Please enter name of model.")

