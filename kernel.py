import os
import random
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from bayes_opt import BayesianOptimization
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Embedding, Dropout, LSTM
from keras.models import Sequential
from keras.optimizers import adam
from keras.preprocessing import sequence

strip_special_chars = re.compile('[^A-Za-z0-9 ]+')


def clean_sentences(string):
    string = string.lower().replace('<br />', ' ')
    return re.sub(strip_special_chars, '', string.lower())


def generate_word_list(input_dir, base_dir, top_words, use_existing_dict=False):
    vocabulary_dict_np_file = 'vocabulary_dict.npy'
    if use_existing_dict and Path(base_dir.joinpath(vocabulary_dict_np_file)).is_file():
        return np.load(base_dir.joinpath(vocabulary_dict_np_file)).item()

    vocabulary_dict = dict()
    vocabulary_dict_freq = dict()

    for i in list(list(input_dir.joinpath('pos').iterdir()) + list(input_dir.joinpath('neg').iterdir())):
        words_in_text = np.array(clean_sentences(i.read_text()).split())
        for word in words_in_text:
            idx = vocabulary_dict_freq.get(word)
            if idx is None:
                vocabulary_dict_freq[word] = 0
            else:
                vocabulary_dict_freq[word] += 1

    vocabulary_dict_freq = sorted(vocabulary_dict_freq.items(), key=lambda x: x[1], reverse=True)

    for index, item in enumerate(vocabulary_dict_freq):
        if index < top_words:
            vocabulary_dict[item[0]] = index
        else:
            break

    np.save(base_dir.joinpath(vocabulary_dict_np_file), vocabulary_dict)
    return vocabulary_dict


def generate_data(input_dir, phase, vocabulary_dict):
    indexes = list(list(input_dir.joinpath('pos').iterdir()) + list(input_dir.joinpath('neg').iterdir()))

    random.shuffle(indexes)

    data_np_file = 'data.npy'
    labels_np_file = 'labels.npy'
    if (Path(phase.joinpath(data_np_file)).is_file()) or (Path(phase.joinpath(labels_np_file)).is_file()):
        return np.load(phase.joinpath(data_np_file)), np.load(phase.joinpath(labels_np_file))
    else:
        data = []
        labels = []

    for i in indexes:
        data.append(generate_word_vec(i.read_text(), vocabulary_dict))

        if 'pos' in i.parent.name:
            labels.append(1)
        else:
            labels.append(0)

    np.save(phase.joinpath(data_np_file), data)
    np.save(phase.joinpath(labels_np_file), labels)

    return data, labels


def generate_word_vec(text, vocabulary_dict):
    word_vec = []
    words_in_text = np.array(clean_sentences(text).split())
    for word in words_in_text:
        idx = vocabulary_dict.get(word)
        if idx is not None:
            word_vec.append(idx)
    return word_vec


def create_tmp_folders():
    if not os.path.exists('tmp'):
        os.makedirs('tmp')
    if not os.path.exists('tmp/train'):
        os.makedirs('tmp/train')
    if not os.path.exists('tmp/test'):
        os.makedirs('tmp/test')


def prepare_data(top_words, max_length):
    create_tmp_folders()

    vocabulary_dict = generate_word_list(Path('data/train'), Path('tmp'), top_words)

    train_features, train_labels = generate_data(Path('data/train'), Path('tmp/train'), vocabulary_dict)
    val_features, val_labels = generate_data(Path('data/test'), Path('tmp/test'), vocabulary_dict)

    train_features = sequence.pad_sequences(train_features, maxlen=max_length)
    val_features = sequence.pad_sequences(val_features, maxlen=max_length)

    return (train_features, train_labels), (val_features, val_labels)


def build_and_evaluate(data, top_words, dropout=0.2, lstm_units=32, fc_hidden=128, lr=3e-4, verbose=False):
    (train_features, train_labels), (val_features, val_labels) = data
    model = Sequential()
    model.add(Embedding(input_dim=top_words, output_dim=lstm_units, input_length=train_features.shape[1]))
    model.add(LSTM(lstm_units))
    model.add(Dense(units=fc_hidden, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=adam(lr), metrics=['accuracy'])

    return model.fit(train_features, train_labels, validation_data=(val_features, val_labels), epochs=100,
                     batch_size=512, verbose=verbose,
                     callbacks=[EarlyStopping(monitor='val_loss', patience=5, mode='auto', baseline=None)])


def plot_history(history):
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def overfit_batch(sample_size=2000):
    top_words = 5000
    (train_features, train_labels), (val_features, val_labels) = prepare_data(top_words=top_words, max_length=10)
    data = (train_features[:sample_size], train_labels[:sample_size]), \
           (val_features[:sample_size], val_labels[:sample_size])
    history = build_and_evaluate(data, top_words)
    plot_history(history)
    return history.history['val_acc'][-1]


# overfit_batch()

def bayesian_opt(data, top_words):
    optimizer = BayesianOptimization(
        f=build_and_evaluate(data, top_words),
        pbounds={'dropout': (0.0, 0.5), 'lstm_units': (32, 500), 'fc_hidden': (32, 256)},
    )

    optimizer.maximize(
        init_points=10,
        n_iter=30,
    )


bayesian_opt(prepare_data(top_words=5000, max_length=500), 5000)
