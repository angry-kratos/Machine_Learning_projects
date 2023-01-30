from locale import normalize
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)

x_test = tf.keras.utils.normalize(x_test, axis=1)

# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(10, activation='softmax'))

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics =['accuracy'])

# model.fit(x_train, y_train, epochs= 3)

# model.save('handwritten.model')


model = tf.keras.models.load_model("handwritten.model")

# loss, accuracy = model.evaluate(x_test, y_test)


# print(loss)
# print(accuracy)


while os.path.isfile(f"handwritten/untitled.png"):
    try:
        img = cv2.imread(f"handwritten/untitled.png")[:, :, 0]
        cropped_image_1 = img[:,:28]
        cropped_image_2 = img[:, 28:]
        plt.imshow(cropped_image_1[0], cmap=plt.cm.binary)
        plt.show
        # c_img1 = np.invert(np.array([cropped_image_1]))
        # c_img2 = np.invert(np.array([cropped_image_2]))
        # prediction1 = model.predict(c_img1)
        # prediction2 = model.predict(c_img2)
        # print(f"this digit is {np.argmax(prediction1)} and {np.argmax(prediction2)}")
        # plt.imshow(img[0], cmap=plt.cm.binary)
        # plt.show()
    except:
        print("error")
