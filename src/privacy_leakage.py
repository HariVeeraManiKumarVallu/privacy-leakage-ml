import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer

def create_model():
    """
    This function creates and compiles a neural network model with differential privacy.
    """
    keras_model = Sequential([
        Dense(128, activation='relu', input_shape=(784,)),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])

    optimizer = DPKerasSGDOptimizer(
        l2_norm_clip=1.0,             # Clip gradients to limit the influence of each data point
        noise_multiplier=1.1,         # Add noise to the gradients for differential privacy
        num_microbatches=1,           # Number of microbatches (usually 1 for simplicity)
        learning_rate=0.15           # Learning rate for gradient descent
    )

    keras_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return keras_model

def train_model():
    """
    This function loads the MNIST dataset, preprocesses it, trains the model with differential privacy,
    and returns the training history.
    """
    # Load MNIST data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Preprocess the data: Flatten and normalize
    x_train = x_train.reshape((60000, 784)).astype('float32') / 255
    x_test = x_test.reshape((10000, 784)).astype('float32') / 255
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    # Create and train the model
    keras_model = create_model()
    history = keras_model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)
    
    print("Model training complete.")


    # Evaluate the model on the test set
    test_loss, test_accuracy = keras_model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    return history

# Run the training function
if __name__ == "__main__":
    history = train_model()
    
    # Visualize the results
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
