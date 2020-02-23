import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

def load_data(test=False):
    """
    if test == True, load test data, else load training data
    """
    FTRAIN = './data/training.csv'
    FTEST = './data/test.csv'
    fname = FTEST if test else FTRAIN
    df = read_csv(fname)

    # convert image of string type to numpy array
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    # drop incomplete data
    df = df.dropna()  

    # convert image pixel values from range(256) to range(1)
    X = np.vstack(df['Image'].values) / 255.  
    X = X.astype(np.float32)

    # reshape X to (m, 96, 96, 1) here m is the number of images in our df. Image has a shape (96, 96)
    X = X.reshape(-1, 96, 96, 1) 

    # only FTRAIN contains target value
    if not test:
        # last column is image in training.csv
        y = df[df.columns[:-1]].values
        #  normalize keypoints to [-1, 1]
        y = (y - 48) / 48  
        # shuffle training data
        X, y = shuffle(X, y, random_state=42)  
        y = y.astype(np.float32)
    else:
        y = None

    return X, y

def plot_history(history):
    # Plot training & validation loss values
    plt.figure(figsize=(10, 10))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig('training_process.png')
    plt.show()
