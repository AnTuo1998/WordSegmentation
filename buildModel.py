import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers import LSTM,Bidirectional,Dense,Embedding,Masking,TimeDistributed
from keras.models import Sequential,load_model
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
import keras
from config import MAX_SEQ_LEN,WORD_DIM,MODEL_PATH,WEIGHT_PATH,FIGURE_PATH

def build_model(isModel=False):
    if isModel:
        model = load_model(MODEL_PATH)
        model.load_weights(WEIGHT_PATH)
        return model

    model = Sequential()    
    from config import WORD_NUM
    print("wordnum",WORD_NUM)

    model.add(Embedding(input_dim=WORD_NUM, output_dim=WORD_DIM))

    lstm = LSTM(units=2 * MAX_SEQ_LEN, input_shape=(MAX_SEQ_LEN, WORD_DIM), 
                return_sequences=True, dropout=0.3)

    model.add(Bidirectional(layer=lstm, merge_mode='ave'))
    # merge_mode means how to connect two vectors

    model.add(TimeDistributed(Dense(1, activation='sigmoid')))

    model.compile(loss='binary_crossentropy', 
                    optimizer='adam', 
                    metrics=['accuracy'])

    model.save(filepath = MODEL_PATH)
    model.summary()
    try:
        plot_model(model, to_file=FIGURE_PATH+'network.png')
    except ImportError:
        pass
    return model