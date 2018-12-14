import os
import pickle
import numpy as np
np.set_printoptions(threshold=np.inf)
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, Activation, Flatten, Dense
from keras.layers import Conv1D, MaxPool1D, Dropout
from keras.models import Model
from keras.models import load_model
from nltk.tokenize import word_tokenize


def load_data():
    global df_data, classes
    data_source = './data/us_county.csv'
    df_data = pd.read_csv(data_source, header=None)

    df_data = df_data.set_index(0)
    df_data = df_data.drop(["time_zone", "web", "ex_image", "ex_image_cap", "seal"], axis=0)

    df_data = df_data.reset_index()

    print("TRAINING LABELS FREQUENCY")
    print(df_data.groupby([0]).count())
    # print(df_data.sample(5))

    classes = df_data[0].unique()


def load_validation_data():
    data_source = './data/validation-sentences.csv'
    validation_data = pd.read_csv(data_source)
    validation_data = validation_data[['property', 'value', 'sentence']]
    validation_data['property'] = le.transform(validation_data['property'].values)
    # print(validation_data.sample(5))
    return validation_data


def encode_labels():
    global transformed_labels, le
    le = LabelEncoder()
    le.fit(classes)
    print("\n{} labels: {}".format(len(le.classes_), list(le.classes_)))
    print("transform: {}".format(le.transform(le.classes_)))
    transformed_labels = le.transform(df_data[0].values)


def split_data():
    global X_train, X_test, y_train, y_test
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_index, test_index = next(sss.split(df_data[1].str.lower(), transformed_labels))
    X_train, X_test = (df_data[1].str.lower())[train_index], (df_data[1].str.lower())[test_index]
    y_train, y_test = transformed_labels[train_index], transformed_labels[test_index]


def vectorize_corpus():
    global tk, train_sequences, test_sequences, transformed_text
    tk = Tokenizer(num_words=None, char_level=True, lower=True, oov_token='UNK')
    # print(np.random.choice(X_train, 20, replace=False))
    temp_df = np.concatenate((X_train, X_test), axis=0)
    transformed_text = tk.fit_on_texts(temp_df)
    # print(tk.word_counts)
    # print(tk.document_count)
    # print(tk.word_index)
    # print(tk.word_docs)
    # Convert string to index
    train_sequences = tk.texts_to_sequences(X_train)
    test_sequences = tk.texts_to_sequences(X_test)


def load_embedding_weights():
    global vocab_size, embedding_weights
    print("Word index: {}".format(tk.word_index))
    vocab_size = len(tk.word_index)
    print("Vocabulary size: {}".format(vocab_size))
    embedding_weights = []
    embedding_weights.append(np.zeros(vocab_size))
    for char, i in tk.word_index.items():
        onehot = np.zeros(vocab_size)
        onehot[i - 1] = 1
        embedding_weights.append(onehot)
    embedding_weights = np.array(embedding_weights)
    print("Embedding shape: {}".format(embedding_weights.shape))


def train_model():
    load_embedding_weights()

    # ================================ MODEL CONSTRUCTIONS
    # Paramenters
    input_size = 1014
    embedding_size = vocab_size
    conv_layers = [[256, 7, 3],
                   [256, 7, 3],
                   [256, 3, -1],
                   [256, 3, -1],
                   [256, 3, -1],
                   [256, 3, 3]]
    fully_connected_layers = [1024, 1024]
    number_of_classes = len(classes)
    dropout_p = 0.5
    optimizer = 'adam'
    loss = 'categorical_crossentropy'

    # Embedding layer initialization
    embedding_layer = Embedding(vocab_size+1, embedding_size,
                                input_length=input_size,
                                weights=[embedding_weights])

    # Model definition
    # Input
    inputs = Input(shape=(input_size,), name='input', dtype='int64')  # shape=(?, 1014)
    # Embedding
    x = embedding_layer(inputs)
    # Conv
    for filter_num, filter_size, pooling_size in conv_layers:
        x = Conv1D(filter_num, filter_size)(x)
        x = Activation('relu')(x)
        if pooling_size != -1:
            x = MaxPool1D(pool_size=pooling_size)(x)
    x = Flatten()(x)

    # Fully connected layers
    for dense_size in fully_connected_layers:
        x = Dense(dense_size, activation='relu')(x)
        x = Dropout(dropout_p)(x)

    # Output layer
    predictions = Dense(number_of_classes, activation='softmax')(x)

    # Build Model
    model = Model(input=inputs, outputs=predictions)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    model.summary()

    model.fit(train_data, to_categorical(train_classes),
              test_data=(test_data, to_categorical(test_classes)),
              batch_size=128, epochs=3, verbose=2)

    # save to file
    # save the model to disk
    filename = 'models/cnn_model.h5'
    model.save(filename)
    return model


def sliding_window(sent_tokens, window_size):
    for i in range(len(sent_tokens) - window_size + 1):
        yield sent_tokens[i: i + window_size]


# ============================ load data
load_data()

# ============================ encode labels
encode_labels()

# ============================ splits on train and test data
split_data()

# ============================ convert string to index
vectorize_corpus()

# Padding
train_data = pad_sequences(train_sequences, maxlen=1014, padding='post')
test_data = pad_sequences(test_sequences, maxlen=1014, padding='post')

# Convert to numpy array
train_data = np.array(train_data, dtype='float32')
test_data = np.array(test_data, dtype='float32')
train_classes = y_train
test_classes = y_test

# load the model from disk
filename = 'models/cnn_model.h5'
exists = os.path.isfile(filename)
if exists:
    model = load_model(filename)
else:
    model = train_model()

# predict samples
validation_data = load_validation_data()

# print("PREDICTIONS ON SENTENCE WINDOWS: {}".format(validation_data.shape))

predictions = []
for index, data in validation_data.iterrows():
    sentence = data[['sentence']].values[0].lower()
    sentence_tokens = word_tokenize(sentence.decode('utf-8'))

    texts = sentence_tokens
    '''texts = []
    for window in sliding_window(sentence_tokens, window_size=2):
        w = ''
        for i in window:
            w += i + " "
        texts.append(w.strip())'''

    texts_sequences = tk.texts_to_sequences(texts)
    pred_data = pad_sequences(texts_sequences, maxlen=1014, padding='post')
    pred_data = np.array(pred_data, dtype='float32')

    y_pred = model.predict(pred_data)

    proba = np.amax(y_pred, axis=1)
    indexes = y_pred.argmax(axis=1)

    bigger_proba = np.argmax(proba)
    class_predicted = indexes[bigger_proba]
    selected_text = texts[bigger_proba]

    predictions.append({'truth': data[['property']].values[0], 'pred': class_predicted,
                        'truth_text': data[['value']].values[0], 'pred_text': selected_text})

    if index < 3:  # data[['property']].values[0] == class_predicted or index < 5:
        print("\n\t==================================================================")
        print("\tSENT TOKENS/WINDOW: {}".format(texts))
        print("\tINDIVIDUAL TOKEN PRED: {}".format(list(indexes)))
        print("\tPROBABILITIES: {}".format(list(proba)))

        print("\tSENTENCE TRUTH: {} - {}".format(data[['property']].values[0], le.inverse_transform(np.array([data[['property']].values[0]]))))
        print("\tFINAL AND HIGHER PREDICTION IN SENTENCE: {} - {}\n".format(class_predicted, le.inverse_transform(np.array([class_predicted]))))

        print("\tTRUTH EXTRACTION: {}".format(data[['value']].values[0]))
        print("\tPRED EXTRACTION: {}".format(selected_text))
        print("\t==================================================================")

print(" ")

count_by_prop = {}
for i in range(len(predictions)):

    if predictions[i]['truth'] == predictions[i]['pred']:

        property_name = le.inverse_transform(np.array([predictions[i]['truth']]))[0]

        if str(property_name) in count_by_prop.keys():
            count_by_prop[str(property_name)] += 1
        else:
            count_by_prop[str(property_name)] = 1

        print("{} = t: {}-{} | p: {}-{}".format(property_name,  predictions[i]['truth'], predictions[i]['truth_text'], predictions[i]['pred'], predictions[i]['pred_text']))

props = validation_data.groupby(['property'], as_index=False).count()['property'].values
props_count = validation_data.groupby(['property'], as_index=False).count()['sentence'].values

dict_validation_count = {}
for i in range(len(props)):
    dict_validation_count[le.inverse_transform(np.array([props[i]]))[0]] = props_count[i]

# print(dict_validation_count)
# print(count_by_prop)

final = {}
for k, v in dict_validation_count.iteritems():
    if k in count_by_prop:
        final[k] = float(count_by_prop[k]) / float(v)
    else:
        final[k] = 0

print("\nPRECISION BY CLASS:")
print(final)

count_total = 0
for k, v in count_by_prop.iteritems():
    count_total += v

print("\nTotal Precision: {}/{} = {}".format(count_total, validation_data.shape[0], count_total / float(validation_data.shape[0])))
