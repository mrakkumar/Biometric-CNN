import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers, losses, initializers

class ModelClass:

    def __init__(self, input_shape, num_classes, seed = None):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.make_model(self.input_shape, self.num_classes, seed)
        self.modelpath = None
        self.config = None
        self.compiled = False

    def make_model(self, input_shape, num_classes, seed = None):
        # I think Mukund experimented with different seed values in the Glorot function and ensembled them
        initializer = initializers.GlorotUniform(seed = seed)
        model = models.Sequential()
        model.add(layers.Conv2D(16, (5, 5), activation = 'relu', input_shape = input_shape, kernel_initializer = initializer))
        model.add(layers.MaxPool2D((2, 2)))
        model.add(layers.Conv2D(64, (5, 5), activation = 'relu'))
        model.add(layers.MaxPool2D((2, 2)))
        model.add(layers.Flatten())
        # Not sure of the number of neurons in the FC layer before the output layer
        # The paper mentions "2 dropout layers", although it doesn't say anything about which ones have dropout attached
        model.add(layers.Dense(6000, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(num_classes))
        model.add(layers.Dropout(0.5))
        model.add(layers.Softmax())

        # Not sure of the hyperparameters in the Adadelta optimizer
        opt = optimizers.Adadelta(learning_rate=0.001, rho=0.95, epsilon=1e-07, name='Adadelta')
        model.compile(optimizer = opt,
                      loss = losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics = ['accuracy'])

        return model


# Testing the code
if __name__ == '__main__':
    # Resolving cuDNN error
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

    mnist = datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    _, rows, cols = train_images.shape

    # Add the "channel" axis to the images
    train_images = tf.expand_dims(train_images, axis = 3)
    test_images = tf.expand_dims(test_images, axis = 3)
    train_images = train_images / 255
    test_images = test_images / 255

    num_epochs = 1
    model_obj = ModelClass((rows, cols, 1), 10)

    ### performance and fitting is a separate entity
    model_obj.model.fit(train_images, train_labels, epochs = num_epochs)
    test_loss, test_acc = model_obj.model.evaluate(test_images, test_labels, verbose = 2)
    print('\nTest accuracy:', test_acc)
