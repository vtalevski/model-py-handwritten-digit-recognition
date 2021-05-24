import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

mnist = tf.keras.datasets.mnist  # Load the dataset.
(x_train, y_train), (x_test, y_test) = mnist.load_data()  # Split the dataset into training set and test set.

x_train = tf.keras.utils.normalize(x_train, axis=1)  # Feature scale the training set.
x_test = tf.keras.utils.normalize(x_test, axis=1)  # Feature scale the test set.

# Artificial Neural Network.
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

# Define the Gradient Descend, the Loss Function and the metrics.
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model.
model.fit(x_train, y_train, epochs=3)

# Calculate the accuracy and the loss.
loss, accuracy = model.evaluate(x_test, y_test)
print(f"The model's accuracy is '{accuracy}'.")
print(f"The model's loss is '{loss}'.")

model.save('model')

for single_image_index in range(0, 19):
    single_image = cv.imread(f'./digits/{single_image_index}.png')[:, :, 0]
    single_image = np.invert(np.array([single_image]))  # The invert method gives a black digit on a white background.
    single_image_prediction = model.predict(single_image)
    print(f"The result is probably a '{np.argmax(single_image_prediction)}'.")
    plt.imshow(single_image[0], cmap=plt.cm.binary)  # The 'cmap=plt.cm.binary' code draws a black and white picture.
    plt.show()
