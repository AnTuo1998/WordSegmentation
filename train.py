import matplotlib.pyplot as plt
import numpy as np
import processData
from config import TRAIN_DATA_PATH,EPOCH_SIZE,BATCH_SIZE
from sklearn.model_selection import train_test_split
from buildModel import build_model
from keras.callbacks import ModelCheckpoint



def myplot(history):
    x = []
    for i in range(1, EPOCH_SIZE + 1):
        x.append(i)
    fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize = (12, 12))
    axes[0].plot(x,history.history['acc'], label = 'train')
    axes[0].plot(x,history.history['val_acc'], label = 'validation')
    axes[0].legend()
    axes[0].grid(axis = 'y')
    axes[0].set_xlabel(xlabel = 'epoch')
    axes[0].set_ylabel(ylabel = 'accuracy')
    axes[0].set_title(label = 'Accuracy')
    axes[1].plot(x,history.history['loss'], label = 'train')
    axes[1].plot(x,history.history['val_loss'], label = 'validation')
    axes[1].legend()
    axes[1].grid(axis = 'y')
    axes[1].set_xlabel(xlabel = 'epoch')
    axes[1].set_ylabel(ylabel = 'loss')
    axes[1].set_title(label = 'Loss')
    plt.savefig("./result.png")

def train():
    X, y, vocabDict = processData.processTrainData(TRAIN_DATA_PATH)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

    model = build_model()

    checkpointer = ModelCheckpoint(filepath="/tmp/weights.hdf5", verbose=1, save_best_only=True)
    history = model.fit(X_train,y_train,validation_data=(X_val,y_val), 
                    epochs=EPOCH_SIZE, batch_size=BATCH_SIZE, callbacks=[checkpointer])
    model.save('word_spliter.mod')
    
    myplot(history)

    
