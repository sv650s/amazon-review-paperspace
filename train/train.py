
# Commented out IPython magic to ensure Python compatibility.
from __future__ import absolute_import, division, print_function, unicode_literals


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding, \
    SpatialDropout1D, Flatten, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import model_to_dot
from tensorflow.keras.initializers import Constant
from tensorflow.keras.models import load_model


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

from psutil import virtual_memory

import pandas as pd
import numpy as np
from datetime import datetime
import os
import logging
import argparse


import util.keras_util as ku
import util.report_util as ru

import random

# this allows us to import util directory from the root of project
import sys
sys.path.append('../')

DATE_FORMAT = '%Y-%m-%d'
TIME_FORMAT = '%Y-%m-%d %H:%M:%S'

# length of our embedding - 300 is standard
EMBED_SIZE = 300
BATCH_SIZE = 32
LEARNING_RATE = 0.01
DROPOUT_RATE = 0.2

# From EDA, we know that 90% of review bodies have 100 words or less,
# we will use this as our sequence length
MAX_SEQUENCE_LENGTH = 100


# used to fix seeds
RANDOM_SEED = 1


# set up logging
LOG_FORMAT = "%(asctime)-15s %(levelname)-7s %(name)s.%(funcName)s" \
    " [%(lineno)d] - %(message)s"
logger = logging.getLogger(__name__)


def check_resources():
    """
    Check what kind of resources we have for training
    :return:
    """

    # checl to make sure we are using GPU here
    tf.test.gpu_device_name()

    # check that we are using high RAM runtime
    ram_gb = virtual_memory().total / 1e9
    print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))

    # if ram_gb < 20:
    #   print('To enable a high-RAM runtime, select the Runtime → "Change runtime type"')
    #   print('menu, and then select High-RAM in the Runtime shape dropdown. Then, ')
    #   print('re-execute this cell.')
    # else:
    #   print('You are using a high-RAM runtime!')


def fix_seed(seed: int):

    # fix random seeds
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def load_data(data_file: str, feature_column:str, label_column: str):

    df = pd.read_csv(data_file)
    reviews = df[feature_column]
    rating = df[label_column]

    # pre-process our lables
    # one hot encode our star ratings since Keras/TF requires this for the labels
    y = OneHotEncoder().fit_transform(rating.values.reshape(len(rating), 1)).toarray()


    # split our data into train and test sets
    reviews_train, reviews_test, y_train, y_test = train_test_split(reviews, y, random_state=1)

    return reviews_train, reviews_test, y_train, y_test

def preprocess_data(feature_train, feature_test, embedding_file: str):

    # Pre-process our features (review body)
    t = Tokenizer(oov_token="<UNK>")
    # fit the tokenizer on the documents
    t.fit_on_texts(feature_train)
    # tokenize both our training and test data
    train_sequences = t.texts_to_sequences(feature_train)
    test_sequences = t.texts_to_sequences(feature_test)

    print("Vocabulary size={}".format(len(t.word_counts)))
    print("Number of Documents={}".format(t.document_count))


    # pad our reviews to the max sequence length
    X_train = sequence.pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    X_test = sequence.pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

    print('Train review vectors shape:', X_train.shape, ' Test review vectors shape:', X_test.shape)

    """## Load our pre-trained embedding

    embeddings_index will be a map where key == word, value == the embedding vector
    """

    EMBEDDING_INDEX_FILE = f'{output_dir}/models/glove.840B.300d-embedding_index'

    embeddings_index = {}

    if os.path.exists(f'{EMBEDDING_INDEX_FILE}.npy'):
      print(f'Loading {EMBEDDING_INDEX_FILE}.npy')
      embeddings_index = np.load(f'{EMBEDDING_INDEX_FILE}.npy',
                                 allow_pickle = True).item()
    else:
      print('Indexing word vectors.')

      with open(embedding_file) as f:
          for line in f:
              word, coefs = line.split(maxsplit=1)
              coefs = np.fromstring(coefs, 'f', sep=' ')
              embeddings_index[word] = coefs
      np.save(EMBEDDING_INDEX_FILE, embeddings_index)

    print(type(embeddings_index))
    print(np.shape(embeddings_index))
    print('Found %s word vectors.' % len(embeddings_index))



    """## Create Embedding Matrix based on our tokenizer

    For every word in our vocabulary, we will look up the embedding vector and add the it to our embedding matrix

    The matrix will be passed in as weights in our embedding layer later

    If there is word that does not exist in the pre-trained embedding vocabulary, we will leave the weights as 0 vector and save off the word into a CSV file later for analysis
    """

    # this is a map with key == word, value == index in the vocabulary
    word_index = t.word_index
    print(f'word_index length: {len(word_index)}')

    # we are going to use the entire vocab so we can alter this from the example
    # num_words = min(MAX_NUM_WORDS, len(word_index) + 1)

    # start with a matrix of 0's
    embedding_matrix = np.zeros((len(word_index) + 1, EMBED_SIZE))

    # if a word doesn't exist in our vocabulary, let's save it off
    missing_words = []
    print(f'embedding_matrix shape: {np.shape(embedding_matrix)}')
    for word, i in word_index.items():
        # print(f'word: {word} i: {i}')
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None and np.shape(embedding_vector)[0] == EMBED_SIZE:
            # words not found in embedding index will be all-zeros.
            # print(f'i: {i} embedding_vector shape: {np.shape(embedding_vector)}')
            embedding_matrix[i] = embedding_vector
        else:
          missing_words.append(word)

    print(f'Number of missing words from our vocabulary: {len(missing_words)}')

    """Save off our missing words into a csv file so we can analyze this later"""

    # save missing words into a file so we can analyze it later
    missing_words_df = pd.DataFrame(missing_words)
    missing_words_df.to_csv(MISSING_WORDS_FILE, index=False)

    """**Build LSTM Model Architecture**"""

    return X_train, X_test, t, embedding_matrix



if __name__ == "__main__":

    check_resources()
    fix_seed(RANDOM_SEED)

    parser = argparse.ArgumentParser()

    # TODO: parameterize entire training
    # parser.add_argument("datafile", help="source data file")
    parser.add_argument("-i", "--input_dir", help="input directory. Default /storage",
                        default="/storage")
    parser.add_argument("-o", "--output_dir", help="output directory. Default /artifacts",
                        default="/artifacts")
    # parser.add_argument("-m", "--modeldir", help="output directory. Default /artifacts/models",
    #                     default="/artifacts/models")
    # parser.add_argument("-r", "--reportdir", help="report directory. Default /artifacts/reports",
    #                     default="/artifacts/reports")


    parser.add_argument("-f", "--feature_column", help="feature column. Default star_rating",
                        default="star_rating")
    parser.add_argument("-l", "--label_column", help="label column. Default label_column",
                        default="review_body")

    parser.add_argument("-s", "--sample_size", help="Sample size (ie, 50k). Default test",
                        default="test")
    parser.add_argument("-p", "--patience", help="patience. Default = 4", default=4)
    parser.add_argument("-c", "--lstm_cells", help="Number of LSTM cells. Default = 128", default=128)
    parser.add_argument("-e", "--epochs", help="Max number epochs. Default = 20", default=20)

    parser.add_argument("-l", "--loglevel", help="log level", default="INFO")

    # parser.add_argument("-d", "--debug", action='store_true',
    #                     help="debug mode", default=False)


    # get command line arguments
    args = parser.parse_args()

    # process argument
    if args.loglevel is not None:
        loglevel = getattr(logging, args.loglevel.upper(), None)
    logging.basicConfig(format=LOG_FORMAT, level=loglevel)
    logger = logging.getLogger(__name__)




    parser = argparse()

    # process argument
    if args.loglevel is not None:
        loglevel = getattr(logging, args.loglevel.upper(), None)
    logging.basicConfig(format=LOG_FORMAT, level=loglevel)
    logger = logging.getLogger(__name__)

    input_dir = args.input_dir
    output_dir = args.output_dir
    label_column = args.label_column
    feature_column = args.feature_column
    lstm_cells = args.lstm_cells
    epochs  = args.epochs
    sample_size = args.sample_size


    debug = False
    if sample_size == "test":
        debug = True


    MODEL_NAME = f"LSTMB{lstm_cells}"
    ARCHITECTURE = f"1x{lstm_cells}"
    DESCRIPTION = f"1 Layer {lstm_cells} LSTM Units, No Dropout, GloVe Embedding (with stop words, nonlemmatized), Balanced Weights"
    FEATURE_SET_NAME = "glove_with_stop_nonlemmatized"
    PATIENCE = 4

    if debug:
      data_file = f'{input_dir}/data/amazon_reviews_us_Wireless_v1_00-test-preprocessed.csv'
      MODEL_NAME = f'test-{MODEL_NAME}'
      MISSING_WORDS_FILE = f'{output_dir}/reports/glove_embedding-missing_words-test.csv'
    else:
      data_file = f"{input_dir}/data/amazon_reviews_us_Wireless_v1_00-{sample_size}-with_stop_nonlemmatized-preprocessed.csv"
      MISSING_WORDS_FILE = f'{output_dir}/reports/glove_embedding-missing_words-{sample_size}.csv'
      # TODO: parameterize this later
      ku.ModelWrapper.set_report_filename('glove_embedding_with_stop_nonlemmatized-dl_prototype-report.csv')


    EMBEDDING_FILE = f'{input_dir}/data/embeddings/glove.840B.300d.txt'


    ku.ModelWrapper.set_reports_dir(f'{output_dir}/reports')
    ku.ModelWrapper.set_models_dir(f'{output_dir}/models')


    reviews_train, reviews_test, y_train, y_test = load_data(data_file, feature_column, label_column)

    X_train, X_test, t, embedding_matrix = preprocess_data(reviews_train, reviews_test, EMBEDDING_FILE)

    vocab_size = len(t.word_index)+1

    # building our network
    model = Sequential()
    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    model.add(Embedding(input_dim=vocab_size,
                                output_dim=EMBED_SIZE,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False))
    # model.add(Embedding(input_dim=vocab_size, output_dim=EMBED_SIZE, input_length=MAX_SEQUENCE_LENGTH))
    model.add(LSTM(lstm_cells, recurrent_dropout=DROPOUT_RATE))
    model.add(Dense(5, activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.001),
                  metrics=["categorical_accuracy"])

    print(model.summary())

    # reduce learning rate if we sense a plateau
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  restore_best_weights=True)

    early_stop = EarlyStopping(monitor='val_loss',
                               patience=PATIENCE,
                               verbose=1,
                               restore_best_weights=True)

    weights = compute_class_weight('balanced', np.arange(1, 6), y_train)
    weights_dict = {i: weights[i] for i in np.arange(0, len(weights))}
    print(f'class weights: {weights}')
    print(f'class weights_dict: {weights_dict}')



    mw = ku.ModelWrapper(model,
                         MODEL_NAME,
                         ARCHITECTURE,
                         FEATURE_SET_NAME,
                         label_column,
                         feature_column,
                         data_file,
                         embed_size=EMBED_SIZE,
                         tokenizer=t,
                         description=DESCRIPTION)

    network_history = mw.fit(X_train, y_train,
                          batch_size=BATCH_SIZE,
                          epochs=epochs,
                          verbose=1,
                          validation_split=0.2,
                          class_weight=weights_dict,
                          callbacks=[early_stop, reduce_lr])

    scores = mw.evaluate(X_test, y_test)
    print("Accuracy: %.2f%%" % (mw.scores[1]*100))

    # pu.plot_network_history(mw.network_history, "categorical_accuracy", "val_categorical_accuracy")
    # plt.show()

    print("\nConfusion Matrix")
    print(mw.confusion_matrix)

    print("\nClassification Report")
    print(mw.classification_report)

    # fig = plt.figure(figsize=(5,5))
    # pu.plot_roc_auc(mw.model_name, mw.roc_auc, mw.fpr, mw.tpr)

    print(f'Score: {ru.calculate_metric(mw.crd)}')

    """**Save off various files**"""

    mw.save(output_dir, append_report=True)

    """# Test That Our Models Saved Correctly"""

    model_loaded = load_model(mw.model_file)
    scores = model_loaded.evaluate(X_test, y_test, verbose=1)
    print("Accuracy: %.2f%%" % (scores[1]*100))

    # this takes too long for real models
    if debug == True:
      y_predict = model_loaded.predict(X_test)
      y_predict_unencoded = ku.unencode(y_predict)
      y_test_unencoded = ku.unencode(y_test)

      # classification report
      print(classification_report(y_test_unencoded, y_predict_unencoded))

      # confusion matrix
      print(confusion_matrix(y_test_unencoded, y_predict_unencoded))

    print(datetime.now())