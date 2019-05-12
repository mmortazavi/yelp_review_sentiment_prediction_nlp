
import os
import warnings
import matplotlib
import time
import pandas as pd

from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from yelp.model import run
from yelp.data import read_json_to_df
from yelp.preprocessing import text_clean
from yelp.utils import target_binning, target_to_categorical
from yelp.preprocessing import tokenize, create_embedding_matrix

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings("ignore", category=DeprecationWarning)

RESULT_PATH = '/Users/majidmortazavi/GitHub/interveiew_excercises/NewYorker/resutls'
DATA_PATH = '/Users/majidmortazavi/data_sciences/case_studies/NewYorker/data/'
DATA_FILE = 'review.json'

EMBEDDING_PATH = '/Users/majidmortazavi/data_sciences/embeddings'
EMBEDDING_VEC = 'glove.6B.100d.txt'
EMBEDDING_DIM = 100

def main():

    print(50*'-')
    print('Reading and Cleaning Data...')
    start = time.time()

    df = read_json_to_df(DATA_PATH, DATA_FILE, chunk_size = 100000)

    # Drop Unnecessary Columns
    df = df.drop(['review_id','user_id','business_id','date','useful','funny','cool'],axis=1)

    df['clean_text'] = df['text'].apply(lambda tx: text_clean(tx))

    end = time.time()
    print('Reading and Cleaning Took {0:0.2f} s.'.format(end-start))

    df = target_binning(df)
    labels, mapping = target_to_categorical(df)

    print(50*'-')
    max_seq_length = 100  #or based on max sequences lenght -> len(max(sequences,key=len))

    print('Tokenizing and Padding (pad_size = {})...'.format(max_seq_length))
    start = time.time()
    sequences, word_index = tokenize(df)

    text_data = pad_sequences(sequences, maxlen=max_seq_length)
    end = time.time()
    print('Tokenizing and Padding Took {0:0.2f} s.'.format(end-start))
    print()

    print(50*'-')
    print('Train, Test Split...')
    start = time.time()

    x_train,x_val, y_train, y_val = train_test_split(text_data, 
                                                    labels, 
                                                    test_size=0.2,
                                                    stratify =labels)

    end = time.time()
    print('Train, Test Split Took {0:0.2f} s.'.format(end-start))

    embedding_matrix = create_embedding_matrix (word_index,
                                                EMBEDDING_PATH,
                                                embedding_vec = EMBEDDING_VEC,
                                                embedding_dim = EMBEDDING_DIM
                                                )

    model = run(x_train,y_train,
                    x_val,y_val,
                    word_index, 
                    EMBEDDING_DIM, 
                    embedding_matrix, 
                    max_seq_length,
                    epochs = 20,
                    batch_size = 256,
                    path_to_fig = RESULT_PATH,
                    target_names = list(mapping))

if __name__ == "__main__":
    main()
