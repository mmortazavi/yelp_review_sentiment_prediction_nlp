import os
import time
import warnings
import pandas as pd
import matplotlib
import matplotlib.pylab as plt

from yelp.preprocessing import tokenize, create_embedding_matrix

from keras.layers import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Bidirectional,GlobalMaxPool1D,Conv1D
from keras.layers import LSTM,Input,Dense,Dropout,Activation
from keras.models import Model

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings("ignore", category=DeprecationWarning)

matplotlib.pyplot.show()

def run(x_train,y_train,
        x_val,y_val,
        word_index, 
        embedding_dim, 
        embedding_matrix, 
        max_seq_length,
        epochs,
        batch_size,
        path_to_fig,
        target_names):

    print(50*'-')
    print('Training Model...')

    start = time.time()

    embedding_layer = Embedding(len(word_index)+1,embedding_dim,weights=[embedding_matrix],input_length=max_seq_length,trainable=False)

    inp = Input(shape=(max_seq_length,))
    x = embedding_layer(inp)
    x = Bidirectional(LSTM(50,return_sequences=True,dropout=0.1,recurrent_dropout=0.1))(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(50,activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(2,activation='sigmoid')(x)
    model = Model(inputs=inp,outputs=x)

    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

    history = model.fit(x_train,y_train,validation_data=(x_val,y_val),epochs= epochs ,batch_size= batch_size)
    end = time.time()
    print('Training Complete in {0:0.2f} s.'.format(end-start))
    print(50*'-')
    print()

    # summarize history for accuracy
    fig1 = plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    fig1.savefig(os.path.join(path_to_fig, 'acc.pdf'), dpi=300, bbox_inches='tight')

    # summarize history for loss
    fig2 = plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    fig2.savefig(os.path.join(path_to_fig, 'loss.pdf'), dpi=300, bbox_inches='tight')
    print('Model Training Hostory Saved in Results Directory.')
    print()

    y_pred = model.predict(x_val)
    y_pred= y_pred.argmax(1)
    y_val =y_val.argmax(1)
    print(50*'-')
    print('Model Performance:')
    print("Score:",round(accuracy_score(y_val,y_pred)*100,2))
    print("Classification Report:\n",classification_report(y_val,y_pred, target_names= target_names))
    print(50*'-')

    return model
