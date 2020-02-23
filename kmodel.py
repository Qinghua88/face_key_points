from keras.models import Sequential
from keras.models import load_model
from keras.layers import Convolution2D, MaxPooling2D, Dropout
from keras.layers import Flatten, Dense
from utils import plot_history
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam

def create_model():
    model = Sequential()
    model.add(Convolution2D(32, (5, 5), input_shape=(96,96,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Convolution2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())

    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(30)) # model will predict 15 keypoints (x, y), so 30 values in total

    return model

def compile_model(model):
    optimizer = 'adam'
    loss = 'mean_squared_error'
    metrics = ['mean_squared_error']
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

def train_model(model, X_train, y_train):
    history = model.fit(X_train, y_train, epochs=200, batch_size=200, verbose=1, validation_split=0.2)
    plot_history(history)
    return history

def save_model(model, fileName):
    model.save(fileName + '.h5')

def load_trained_model(fileName):
    return load_model(fileName + '.h5')
