import os
import math
import numpy as np
from copy import deepcopy
np.set_printoptions(threshold=np.inf)
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import resample
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, Activation, Flatten, Dense
from keras.layers import Conv1D, MaxPool1D, Dropout
from keras.models import Model
from keras.models import load_model
from nltk.tokenize import word_tokenize


def load_validation_data(le, class_):
    data_source = './data/validation-sentences.csv'
    validation_data = pd.read_csv(data_source)
    validation_data = validation_data[['property', 'value', 'sentence']]
    validation_data.loc[validation_data['property'] != class_, 'property'] = 'other'
    validation_data['property'] = le.transform(validation_data['property'].values)
    # print(validation_data.sample(5))
    return validation_data


def load_data():
    data_source = './data/us_county.csv'
    df_data = pd.read_csv(data_source, header=None, encoding="utf-8")

    df_data = df_data.set_index(0)
    df_data = df_data.drop(["time_zone", "web", "ex_image", "ex_image_cap", "seal"], axis=0)

    df_data = df_data.reset_index()
    print("\nDESCRIBE")
    print(df_data.describe())
    print("\nTRAINING DATA FREQUENCY")
    print(df_data.groupby([0]).count())
    print("\n")

    df_tokens = remove_outliers(df_data)

    print("\nDESCRIBE")
    print(df_tokens.describe())
    print("\nTRAINING DATA FREQUENCY")
    print(df_tokens.groupby([0]).count())
    print("\n")
    df_max = df_tokens.groupby([0])[2].max().reset_index()
    print("\n[NEW] max tokens count")
    print(df_max)
    #print("\nMean count of tokens")
    #print(df_tokens.groupby([0])[2].mean().round(0).astype(int))

    return df_data, df_data[0].unique(), df_max  # returns data and classes


def remove_outliers(df_data):
    count_tokens = []
    for i, d in df_data.iterrows():
        d2 = len(word_tokenize(d[1]))
        count_tokens.append([d[0], d[1], d2])  # [property, value, tokens count]

    df_tokens = pd.DataFrame(count_tokens)

    df_max = df_tokens.groupby([0])[2].max().reset_index()
    print("\nMax tokens count")
    print(df_max)

    df_mean = df_tokens.groupby([0])[2].mean().reset_index()
    print("\nMean tokens count")
    print(df_mean)

    df_std = df_tokens.groupby([0])[2].std().reset_index()
    print("\nStd tokens count")
    print(df_std)

    properties_name = np.array(df_std[0].values)
    bound = np.array(df_mean[2].values) + np.array(df_std[2].values)
    df_bounds = pd.DataFrame(data=list(zip(properties_name.T, bound.T)), columns=['props', 'bound'])

    print("\nBounds")
    print(df_bounds)
    for i, d in df_tokens.iterrows():
        prop = d[0]
        tuple = df_bounds[df_bounds['props'] == prop]
        prop_bound = math.ceil(tuple.iloc[0, 1])
        if d[2] > prop_bound:
            # print("prop name: {}\nsent tokens: {}\nprop bound: {}".format(prop, d[2], prop_bound))
            df_tokens.drop([i], inplace=True)

    return df_tokens


def rewrite_labels2other(data, class_):
    data.loc[data[0] != class_, 0] = 'other'
    return data


def downsampling(df):
    df_class_f = df[df[0] == 'other']
    df_class_t = df[df[0] != 'other']

    majority = None
    minority = None
    if df_class_t.shape[0] > df_class_f.shape[0]:  # downsample class t
        majority = df_class_t
        minority = df_class_f
    elif df_class_f.shape[0] > df_class_t.shape[0]:  # downsample class f
        majority = df_class_f
        minority = df_class_t
    elif df_class_f.shape[0] == df_class_t.shape[0]:
        return df

    # Downsample majority class
    if majority is not None and minority is not None:
        print("Majority samples size: %s" % majority.shape[0])
        print("Minority samples size: %s" % minority.shape[0])

        majority_downsampled = resample(majority,
                                        replace=False,
                                        n_samples=minority.shape[0],
                                        random_state=42)

        print("After downsampling: %s" % majority_downsampled.shape[0])

        df_downsampled = pd.concat([majority_downsampled, minority])
        return df_downsampled

    return df


def encode_labels(data):
    le = LabelEncoder()
    le.fit(data[0].unique())
    print("\n{} labels: {}".format(len(le.classes_), list(le.classes_)))
    print("\ntransform: {}".format(le.transform(le.classes_)))
    transformed_labels = le.transform(data[0].values)
    return le, transformed_labels


def split_data(data, transformed_labels):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_index, test_index = next(sss.split(data[1].str.lower(), transformed_labels))
    return data[1][train_index], data[1][test_index], transformed_labels[train_index], transformed_labels[test_index]


def vectorize_corpus(X_train, X_test):
    tk = Tokenizer(num_words=None, char_level=True, lower=True, oov_token='UNK')

    temp_df = np.concatenate((X_train, X_test), axis=0)
    tk.fit_on_texts(temp_df)

    train_seq = tk.texts_to_sequences(X_train)
    test_seq = tk.texts_to_sequences(X_test)

    return tk, train_seq, test_seq


def load_embedding_weights(tk):
    print("\nWord index: {}".format(tk.word_index))
    vocab_size = len(tk.word_index)
    print("\nVocabulary size: {}".format(vocab_size))

    embedding_weights = []
    embedding_weights.append(np.zeros(vocab_size))

    for char, i in tk.word_index.items():
        onehot = np.zeros(vocab_size)
        onehot[i - 1] = 1
        embedding_weights.append(onehot)

    embedding_weights = np.array(embedding_weights)

    print("\nEmbedding shape: {}".format(embedding_weights.shape))

    return vocab_size, embedding_weights


def train_model(prop_name, tk, classes, class_weight, train_data, test_data, train_classes, test_classes):
    vocab_size, embedding_weights = load_embedding_weights(tk)

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

    print('\nnumber_of_classes: {}'.format(number_of_classes))

    dropout_p = 0.5
    optimizer = 'adam'
    #loss = 'categorical_crossentropy'
    loss = 'binary_crossentropy'

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

    print("train_data: {}".format(train_data.shape))
    print("test_data: {}".format(test_data.shape))
    print("train_classes: {}".format(len(train_classes)))
    print("test_classes: {}".format(len(test_classes)))

    model.fit(train_data, to_categorical(train_classes),
              validation_data=(test_data, to_categorical(test_classes)),
              batch_size=128, epochs=8, class_weight=class_weight)

    # save to file
    # save the model to disk
    filename = 'models/cnn_model-' + prop_name + '.h5'
    model.save(filename)
    return model


def train_models(data, classes):
    models = []
    for prop_name in classes:
        prop_data = deepcopy(data)

        print("\nTRAINING MODEL FOR {}".format(prop_name))

        # transform all remaining classes to 'other' label
        prop_data = rewrite_labels2other(prop_data, prop_name)

        prop_data = downsampling(prop_data)

        le, transformed_labels = encode_labels(prop_data)

        labels_frequency = prop_data.groupby([0]).count()
        print("\nlabels frequency: {}".format(labels_frequency))

        # DEFINE CLASS WEIGHTS
        class_weights = define_class_weights(labels_frequency, le)

        print("\nclass_weights: {}".format(class_weights))

        X_train, X_test, y_train, y_test = split_data(prop_data, transformed_labels)

        tk, train_seq, test_seq = vectorize_corpus(X_train, X_test)

        # Padding
        train_data = np.array(pad_sequences(train_seq, maxlen=1014, padding='post'), dtype='float32')
        test_data = np.array(pad_sequences(test_seq, maxlen=1014, padding='post'), dtype='float32')

        # load model from disk if it exists
        filename = 'models/cnn_model-' + prop_name + '.h5'
        exists = os.path.isfile(filename)
        if exists:
            print("\nLOADING ALREADY EXISTING MODEL")
            model = load_model(filename)
            models.append([prop_name, le, tk, model])
        else:
            print("\nSTART TRAINING MODEL")
            model = train_model(prop_name, tk, prop_data[0].unique(), class_weights, train_data, test_data, y_train, y_test)
            models.append([prop_name, le, tk, model])
    return models


def define_class_weights(labels_frequency, le):
    class_weights = {}
    labels_freq = []
    for name, group in labels_frequency.iterrows():
        labels_freq.append([name, group[1]])
    if labels_freq[0][1] == labels_freq[1][1]:  # if frequency for both labels is equal
        class_weights[0] = 1
        class_weights[1] = 1
    elif labels_freq[0][1] > labels_freq[1][1]:  # verifies if frequency of first label is bigger than second
        class_weights[le.transform([labels_freq[0][0]])[0]] = 1  # if higher frequency, minor weight
        # if minor frequency, bigger weight (minor label frequency divided by bigger label freq)
        class_weights[le.transform([labels_freq[1][0]])[0]] = round(labels_freq[0][1] / float(labels_freq[1][1]), 2)
    else:  # otherwise
        class_weights[le.transform([labels_freq[0][0]])[0]] = round(labels_freq[1][1] / float(labels_freq[0][1]), 2)
        class_weights[le.transform([labels_freq[1][0]])[0]] = 1
    return class_weights


def sliding_window(sent_tokens, window_size):
    if len(sent_tokens) >= window_size:
        for i in range(len(sent_tokens) - window_size + 1):
            yield sent_tokens[i: i + window_size]
    else:
        yield sent_tokens


def extraction_pipeline(class_, data, tk, le, model, df_max_tokens):

    predictions = []
    for index, d in data.iterrows():
        sentence = d[['sentence']].values[0].lower()
        property_name = d[['property']].values[0]
        #print('{}: {}'.format(property_name, sentence))
        sentence_tokens = word_tokenize(sentence.decode('utf-8'))

        # define window size based on max tokens per property
        window_size = 1
        #if property_name in df_max_tokens[0]:
        all_props = df_max_tokens[0].values
        if property_name in all_props:
            max_len = df_max_tokens.loc[df_max_tokens[0] == property_name, 2]
            window_size = max_len

        #texts = sentence_tokens
        texts = []
        for window in sliding_window(sentence_tokens, window_size=window_size):
            w = ''
            for i in window:
                w += i + " "
            texts.append(w.strip())

        texts_sequences = tk.texts_to_sequences(texts)
        pred_data = pad_sequences(texts_sequences, maxlen=1014, padding='post')
        pred_data = np.array(pred_data, dtype='float32')

        y_pred = model.predict(pred_data)

        proba = np.amax(y_pred, axis=1)
        indexes = y_pred.argmax(axis=1)

        bigger_proba = np.argmax(proba)
        class_predicted = indexes[bigger_proba]
        selected_text = texts[bigger_proba]

        predictions.append({'truth': d[['property']].values[0], 'pred': class_predicted,
                            'truth_text': d[['value']].values[0], 'pred_text': selected_text})

        if index < 3:  # data[['property']].values[0] == class_predicted or index < 5:
            print("\n\t==================================================================")
            print("\tSENT TOKENS/WINDOW: {}".format(texts))
            print("\tINDIVIDUAL TOKEN PRED: {}".format(list(indexes)))
            print("\tPROBABILITIES: {}".format(list(proba)))

            print("\tSENTENCE TRUTH: {} - {}".format(d[['property']].values[0],
                                                     le.inverse_transform(np.array([d[['property']].values[0]]))))
            print("\tFINAL AND HIGHER PREDICTION IN SENTENCE: {} - {}\n".format(class_predicted, le.inverse_transform(
                np.array([class_predicted]))))

            print("\tTRUTH EXTRACTION: {}".format(d[['value']].values[0]))
            print("\tPRED EXTRACTION: {}".format(selected_text))
            print("\t==================================================================")

    print(" ")

    count_by_prop = {}
    for i in range(len(predictions)):

        if predictions[i]['truth'] == predictions[i]['pred']:

            property_name = le.inverse_transform(np.array([predictions[i]['truth']]))[0]
            #print(property_name)
            if str(property_name) in count_by_prop.keys():
                count_by_prop[str(property_name)] += 1
            else:
                count_by_prop[str(property_name)] = 1

            if class_ == property_name:
                print("{} = t: {}-{} | p: {}-{}".format(
                    property_name, predictions[i]['truth'], predictions[i]['truth_text'],
                    predictions[i]['pred'], predictions[i]['pred_text']))

    props = data.groupby(['property'], as_index=False).count()['property'].values
    props_count = data.groupby(['property'], as_index=False).count()['sentence'].values

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

    print("\nTotal Precision: {}/{} = {}".format(count_total, data.shape[0],
                                                 count_total / float(data.shape[0])))


data, classes, df_max_tokens = load_data()

models = train_models(data, classes)

# predict samples
for m in models:
    prop_name, le, tk, model = m
    val_data = load_validation_data(le, prop_name)
    extraction_pipeline(prop_name, val_data, tk, le, model, df_max_tokens)
