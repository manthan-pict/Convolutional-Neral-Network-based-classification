import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
import keras


# trX - training set, each row represents a sample
# trY - training labels, 0 and 1 represent class 'tshirt' and class 'trouser' respectively
# tsX - testing set, each row represents a sample
# tsY - testing labels, 0 and 1 represent class 'tshirt' and class 'trouser' respectively

# Methos to plot graph on the given arrays on x and y axis
def plot_graph(data, name_of_graph, x_axis, y_axis):
    # plt.subplot(1, 2, 1)
    plt.plot(data)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(name_of_graph)
    plt.legend()
    plt.show()


# Normalize the data to bring it in the range of 0-1.
def normalize_data(data):
    data = data / 255
    return data


# Labels encoder using one-hot vector encoding.
def encode_data(data):
    return LabelBinarizer().fit_transform(data)


# CNN model as per project specificspecifications.
def CNN_model():
    model = keras.Sequential([
        keras.layers.Conv2D(64, 5, strides=1, padding='same',
                            activation='relu',
                            input_shape=(32, 32, 3)),
        keras.layers.MaxPooling2D((2, 2), strides=2),
        keras.layers.Conv2D(64, 5, padding='same',
                            activation='relu'),
        keras.layers.MaxPooling2D((2, 2), strides=2),

        keras.layers.Conv2D(128, 5, strides=1, padding='same',
                            activation='relu'),

        keras.layers.Flatten(),
        keras.layers.Dense(3072, activation='relu'),
        keras.layers.Dense(2048, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])
    model.summary()
    return model


class Data:
    trX = io.loadmat('/content/drive/MyDrive/masters/train_32x32.mat')['X']
    trY = io.loadmat('/content/drive/MyDrive/masters/train_32x32.mat')['y']
    tsX = io.loadmat('/content/drive/MyDrive/masters/test_32x32.mat')['X']
    tsY = io.loadmat('/content/drive/MyDrive/masters/test_32x32.mat')['y']

    training_sample = np.array(trX)
    training_labels = np.array(trY)

    test_samples = np.array(tsX)
    test_labels = np.array(tsY)

    # Data Normalization -
    print("Before Normalizing the data")
    print('Minimum possible: {}, Maximum : {}'.format(training_sample.min(), training_sample.max()))
    training_sample = normalize_data(training_sample)
    test_samples = normalize_data(test_samples)
    print("After Normalizing the data")
    print('Minimum possible: {}, Maximum : {}'.format(training_sample.min(), training_sample.max()))

    # encode_data
    training_labels = encode_data(training_labels)
    test_labels = encode_data(test_labels)

    # CNN model
    cnn_model = CNN_model()
    result = cnn_model.fit(training_sample, training_labels, epochs=20, validation_data=(test_samples, test_labels))

    training_accuracy = result.history['accuracy']
    testing_accuracy = result.history['val_accuracy']
    training_loss = result.history['loss']
    testing_loss = result.history['val_loss']

    plot_graph(training_accuracy, "Training Accuracy", "Number of Epoch", "Accuracy" )
    plot_graph(testing_accuracy, "Testing Accuracy", "Number of Epoch", "Accuracy")
    plot_graph(training_loss, "Training Loss", "Number of Epoch", "Loss")
    plot_graph(testing_accuracy, "Testing Loss", "Number of Epoch", "Loss")

    # Evaluate model on test data
    test_loss, test_acc = cnn_model.evaluate(x=test_samples, y=test_labels, verbose=0)

    print('Test accuracy is: {:0.4f} \nTest loss is: {:0.4f}'.
          format(test_acc, test_loss))
