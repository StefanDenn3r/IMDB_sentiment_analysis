import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
from keras.callbacks import EarlyStopping
from keras.datasets import imdb
from keras.layers import Dense, Embedding, Dropout, LSTM
from keras.models import Sequential
from keras.optimizers import adam
from keras.preprocessing import sequence


def prepare_data(max_features, max_length):
    (x_train, y_train), (x_val, y_val) = imdb.load_data(path="imdb.npz",
                                                        num_words=max_features,
                                                        skip_top=0,
                                                        maxlen=None,
                                                        seed=113,
                                                        start_char=1,
                                                        oov_char=2,
                                                        index_from=3)

    x_train = sequence.pad_sequences(x_train, maxlen=max_length)
    x_val = sequence.pad_sequences(x_val, maxlen=max_length)

    return (x_train, y_train), (x_val, y_val)


def build_and_evaluate(data, max_features, dropout=0.2, lstm_units=32, fc_hidden=128, lr=3e-4, verbose=False):
    (x_train, y_train), (x_val, y_val) = data
    model = Sequential()
    model.add(Embedding(input_dim=max_features, output_dim=lstm_units, input_length=x_train.shape[1]))
    model.add(LSTM(lstm_units))
    model.add(Dense(units=fc_hidden, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=adam(lr), metrics=['accuracy'])

    return model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100,
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
    max_features = 5000
    (x_train, y_train), (x_val, y_val) = prepare_data(max_features=max_features, max_length=10)
    data = (x_train[:sample_size], y_train[:sample_size]), (x_val[:sample_size], y_val[:sample_size])
    history = build_and_evaluate(data, max_features, verbose=True)
    plot_history(history)
    return history.history['val_acc'][-1]


def bayesian_opt(data, max_features):
    optimizer = BayesianOptimization(
        f=build_and_evaluate(data, max_features),
        pbounds={'dropout': (0.0, 0.5), 'lstm_units': (32, 500), 'fc_hidden': (32, 256)},
    )

    optimizer.maximize(
        init_points=10,
        n_iter=30,
    )


# overfit_batch()
bayesian_opt(prepare_data(max_features=5000, max_length=500), 5000)
