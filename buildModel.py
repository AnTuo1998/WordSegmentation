import matplotlib.pyplot as plt
from keras.layers import LSTM,Bidirectional,Dense,Embedding,Masking,TimeDistributed
from keras.models import Sequential,load_model
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
import keras
from config import MAX_SEQ_LEN,WORD_DIM,HIDDEN_NUM, DROPOUT, \
                    MODEL_PATH,FIGURE_PATH,HIDDEN_NUM

def build_model(modelName: str) -> keras.models:
    """build or load a model from the name"""
    try:
        model = load_model(MODEL_PATH+modelName+'.hdf5')
        return model,True
    except (NameError,OSError):
        pass

    model = Sequential() 
    from config import WORD_NUM

    model.add(Embedding(input_dim=WORD_NUM, output_dim=WORD_DIM))

    lstm1 = LSTM(2 * HIDDEN_NUM, input_shape=(MAX_SEQ_LEN, WORD_DIM), 
                return_sequences=True, dropout=DROPOUT)

    model.add(Bidirectional(layer=lstm1, merge_mode='ave'))
    # merge_mode means how to connect two vectors
    lstm2 = LSTM(units=4 * HIDDEN_NUM,return_sequences=True, dropout=DROPOUT)
    
    model.add(Bidirectional(layer=lstm2, merge_mode='ave'))

    lstm3 = LSTM(units=4 * HIDDEN_NUM,return_sequences=True, dropout=DROPOUT)
    
    model.add(Bidirectional(layer=lstm3, merge_mode='ave'))
    
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))

    model.compile(loss='binary_crossentropy', 
                    optimizer='adam', 
                    metrics=['accuracy'])

    model.save(filepath = MODEL_PATH+modelName+'.mod')
    model.summary()
    
    plot_model(model, to_file=FIGURE_PATH+modelName+'.network.png',show_shapes=True)
    
    return model,False