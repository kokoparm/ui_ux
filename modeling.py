import glob
import os

import numpy as np
from keras.callbacks import History
from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


class KSLtoText:
    def __init__(self):
        self.data = []
        self.model = Sequential()
        self.history = History()
        self.dense_size = 0
        self.epochs = 0
        self.batch_size = 0
        self.npy = False

    def set_train_test(self, categories, base_dir, img_size=100, npy=False):
        self.npy = npy
        X = []
        Y = []
        self.dense_size = len(categories)
        for index, cat in enumerate(categories):
            # print(index,cat)
            files = glob.glob(os.path.join(base_dir, cat, "*.*"))
            # print(files)
            for f in files:
                if not npy:
                    img = img_to_array(
                        load_img(f, color_mode="rgb", target_size=(img_size, img_size))
                    )
                else:
                    img = np.load(f)
                X.append(img)
                Y.append(index)
        X = np.asarray(X)
        Y = np.asarray(Y)
        if not npy:
            X = X.astype("float32") / 255.0
        Y = to_categorical(Y, self.dense_size)
        self.data = train_test_split(X, Y, test_size=0.2, random_state=1)
        print(self.data[3])

    def set_model(self):
        if self.npy:
            self.model.add(Input(shape=self.data[0].shape[1:]))
        else:
            self.model.add(Conv2D(100, (3, 3), padding="same", input_shape=self.data[0].shape[1:]))
            self.model.add(Activation("relu"))
            self.model.add(Conv2D(64, (3, 3)))
            self.model.add(Activation("relu"))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            self.model.add(Dropout(0.25))

            self.model.add(Conv2D(64, (3, 3), padding="same"))
            self.model.add(Activation("relu"))
            self.model.add(Conv2D(64, (3, 3)))
            self.model.add(Activation("relu"))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation("relu"))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.dense_size))
        self.model.add(Activation("softmax"))

        self.model.summary()

    def train_model(self, epochs, batch_size):
        self.epochs = epochs
        self.batch_size = batch_size
        self.model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])
        self.history = self.model.fit(
            self.data[0],
            self.data[2],
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(self.data[1], self.data[3]),
        )

    def predict_save(self, filename):
        predict_classes = self.model.predict_classes(self.data[1])
        # prob = self.model.predict_proba(self.data[1])

        self.model.save(f"./models/{filename}_epochs-{self.epochs}_batch-{self.batch_size}.hdf5")
        predict_classes = self.model.predict_classes(self.data[1], batch_size=5)
        true_classes = np.argmax(self.data[3], 1)

        print(confusion_matrix(true_classes, predict_classes))

        print(self.model.evaluate(self.data[1], self.data[3]))


if __name__ == "__main__":
    ktt = KSLtoText()
    categories = [
        "ㄱ",
        "ㄴ",
        "ㄷ",
        "ㄹ",
        "ㅁ",
        "ㅂ",
        "ㅅ",
        "ㅇ",
        "ㅈ",
        "ㅊ",
        "ㅋ",
        "ㅌ",
        "ㅍ",
        "ㅎ",
        "ㅏ",
        "ㅐ",
        "ㅑ",
        "ㅓ",
        "ㅔ",
        "ㅕ",
        "ㅗ",
        "ㅛ",
        "ㅜ",
        "ㅠ",
        "ㅡ",
        "ㅣ",
    ]
    print(len(categories))
    # ktt.set_train_test(categories, "./dataset/captures")
    ktt.set_train_test(categories, "./dataset/keypoints", npy=True)
    print(ktt.data[0].shape[1:])
    print(ktt.data[0][0])
    ktt.set_model()
    ktt.train_model(epochs=500, batch_size=10)
    ktt.predict_save("ksl_255_")
