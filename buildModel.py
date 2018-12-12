import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers import LSTM,Bidirectional,Dense,Embedding,Masking,TimeDistributed
from keras.models import Sequential,load_model
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from config import MAX_SEQ_LEN,WORD_DIM,MODEL_PATH,WORD_NUM

def build_model(model_path=None):
    if model_path != None:
        return load_model(filepath=model_path)

    model = Sequential()    
    model.add(Embedding(input_dim=WORD_NUM, output_dim=WORD_DIM))

    lstm = LSTM(units=2 * MAX_SEQ_LEN, input_shape=(MAX_SEQ_LEN, WORD_DIM), 
                return_sequences=True, dropout=0.4)

    model.add(Bidirectional(layer=lstm, merge_mode='ave'))
    # merge_mode means how to connect two vectors

    model.add(TimeDistributed(Dense(1, activation='sigmoid')))

    model.compile(loss='binary_crossentropy', 
                    optimizer='adam', 
                    metrics=['accuracy'])

    model.save(filepath = MODEL_PATH)
    model.summary()
    plot_model(model, to_file='network.png')
    return model