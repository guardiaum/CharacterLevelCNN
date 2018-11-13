import numpy as np
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

# ============================ load data
data_source = './data/us_county.csv'

df_data = pd.read_csv(data_source, header=None)

classes = df_data[0].unique()
number_of_classes = len(classes)

# ============================ encode labels
le = LabelEncoder()
le.fit(classes)
print(le.classes_)
transformed_labels = le.transform(df_data[0].values)

# ============================ splits on train and test data

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)

train_index, test_index = next(sss.split(df_data[1].values, transformed_labels))

X_train, X_test = (df_data[1].values)[train_index], (df_data[1].values)[test_index]
y_train, y_test = transformed_labels[train_index], transformed_labels[test_index]

# ============================ convert string to index
tk = Tokenizer(num_words=None, char_level=True, lower=True, oov_token='UNK')
transformed_text = tk.fit_on_texts(X_train)

# print(tk.word_counts)
# print(tk.document_count)
# print(tk.word_index)
# print(tk.word_docs)

# construct a new vocabulary
'''
alphabet = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
char_dict = {}
for i, char in enumerate(alphabet):
    char_dict[char] = i + 1

# Use char_dict to replace the tk.word_index
tk.word_index = char_dict.copy()
# Add 'UNK' to the vocabulary
tk.word_index[tk.oov_token] = max(char_dict.values()) + 1
'''

# Convert string to index
train_sequences = tk.texts_to_sequences(X_train)
test_sequences = tk.texts_to_sequences(X_test)

# Padding
train_data = pad_sequences(train_sequences, maxlen=1014, padding='post')
test_data = pad_sequences(test_sequences, maxlen=1014, padding='post')

# Convert to numpy array
train_data = np.array(train_data, dtype='float32')
test_data = np.array(test_data, dtype='float32')
train_classes = y_train
test_classes = y_test

print(tk.word_index)

vocab_size = len(tk.word_index)
print(vocab_size)

embedding_weights = []
embedding_weights.append(np.zeros(vocab_size))

for char, i in tk.word_index.items():
    onehot = np.zeros(vocab_size)
    onehot[i - 1] = 1
    embedding_weights.append(onehot)

embedding_weights = np.array(embedding_weights)
print(embedding_weights.shape)
print(embedding_weights)


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
number_of_classes = number_of_classes
dropout_p = 0.5
optimizer = 'adam'
loss = 'categorical_crossentropy'

# Embedding layer initialization
embedding_layer = Embedding(vocab_size+1, embedding_size,
                            input_length=input_size,
                            weights=[embedding_weights])

# Model definition
# Input
inputs = Input(shape=(input_size,), name='input', dtype='int64') # shape=(?, 1014)
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
          validation_data=(test_data,
                           to_categorical(test_classes)),
          batch_size=128, epochs=10, verbose=2)
