import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import Sequential
from keras.layers import Dense


class NetworkModel:
    def __init__(self):
        self.data = NetworkModel.read_data()
        self.dummies_for_categorical_data()

        all_labels = list(self.data.columns.values)
        y_values = self.data[["pred"]]
        l_x = list(filter(lambda x: x != 'pred', all_labels))
        self.labels = l_x
        self.x_values = self.data[l_x]
        self.y_values = np.ravel(y_values)

        self.scaler = MinMaxScaler()
        self.scaler.fit(self.x_values)
        self.x_values = self.normalize_data(self.x_values)

        self.model = self.prepare_model()

    def prepare_model(self):
        model = Sequential()
        model.add(Dense(8, input_dim=22, activation='relu'))
        model.add(Dense(5, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(self.x_values, self.y_values, epochs=100, batch_size=10, verbose=0)
        scores = model.evaluate(self.x_values, self.y_values)
        print(scores)
        return model

    def predict(self, x):
        print(x)
        x_prep = self.prepare_data(x)
        x_norm = self.normalize_data(x_prep)
        print(x_norm)
        return self.model.predict(x_norm)

    def prepare_data(self, data):
        for k, v in data.items():
            data[k] = float(v)

        return np.array([float(data[label]) for label in self.labels]).reshape(1, -1)

    def normalize_data(self, values):
        return self.scaler.transform(values)

    def dummies_for_categorical_data(self):
        dummies = pd.get_dummies(self.data["cp"], prefix="cp")
        self.data = self.data.join(dummies)
        self.data = self.data.rename(columns={"cp_1.0": "cp_1", "cp_2.0": "cp_2", "cp_3.0": "cp_3", "cp_4.0": "cp_4"})
        del self.data["cp"]

        dummies = pd.get_dummies(self.data["restecg"], prefix="restecg")
        self.data = self.data.join(dummies)
        self.data = self.data.rename(
            columns={"restecg_0.0": "restecg_0", "restecg_1.0": "restecg_1", "restecg_2.0": "restecg_2"})
        del self.data["restecg"]

        dummies = pd.get_dummies(self.data["slope"], prefix="slope")
        self.data = self.data.join(dummies)
        self.data = self.data.rename(columns={"slope_1.0": "slope_1", "slope_2.0": "slope_2", "slope_3.0": "slope_3"})
        del self.data["slope"]

        dummies = pd.get_dummies(self.data["thal"], prefix="thal")
        self.data = self.data.join(dummies)
        self.data = self.data.rename(columns={"thal_3.0": "thal_3", "thal_6.0": "thal_6", "thal_7.0": "thal_7"})
        del self.data["thal"]

    @staticmethod
    def read_data():
        columns = ["age", "sex", "cp", "restbp", "chol", "fbs", "restecg",
                   "thalach", "exang", "oldpeak", "slope", "ca", "thal", "pred"]
        data = pd.read_table("heart_disease_data.csv", sep=',', header=None, names=columns)

        data["pred"].replace(to_replace=[1, 2, 3, 4], value=1, inplace=True)

        return data


if __name__ == '__main__':
    nm = NetworkModel()
    model = nm.prepare_model()