import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


class HeartModel:
    def __init__(self):
        self.data = HeartModel.read_data()
        self.dummies_for_categorical_data()

        self.stdcols = ["restbp", "chol", "thalach", "oldpeak"]

        self.mean_dict = dict((key, self.data[key].mean()) for key in self.stdcols)
        self.std_dict = dict((key, self.data[key].std()) for key in self.stdcols)
        self.age_max = 100.0
        self.ca_max = 3.0

        self.model = self.random_forest_model()

    def dummies_for_categorical_data(self):
        dummies = pd.get_dummies(self.data["cp"], prefix="cp")
        self.data = self.data.join(dummies)
        self.data = self.data.rename(columns={"cp_1.0": "cp_1", "cp_2.0": "cp_2", "cp_3.0": "cp_3", "cp_4.0": "cp_4"})
        del self.data["cp"]

        dummies = pd.get_dummies(self.data["restecg"], prefix="restecg")
        self.data = self.data.join(dummies)
        self.data = self.data.rename(columns={"restecg_0.0": "restecg_0", "restecg_1.0": "restecg_1", "restecg_2.0": "restecg_2"})
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
    def normalize_data(data):
        data_norm = data.copy()
        stdcols = ["restbp", "chol", "thalach", "oldpeak"]
        data_norm[stdcols] = data_norm[stdcols].apply(lambda x: (x-x.mean())/x.std())
        data_norm["age"] = data_norm["age"].apply(lambda x: x/100.0)
        data_norm["ca"] = data_norm["ca"].apply(lambda x: x/3.0)
        return data_norm

    @staticmethod
    def read_data():
        columns = ["age", "sex", "cp", "restbp", "chol", "fbs", "restecg",
                   "thalach", "exang", "oldpeak", "slope", "ca", "thal", "pred"]
        data = pd.read_table("heart_disease_data.csv", sep=',', header=None, names=columns)

        data["pred"].replace(to_replace=[1, 2, 3, 4], value=1, inplace=True)

        return data

    def rl_model(self):
        data_norm = self.normalize_data(self.data)

        labels = list(data_norm.columns.values)

        y_values = data_norm[["pred"]]

        l_x = list(filter(lambda x: x != 'pred', labels))

        x_values = data_norm[l_x]

        rl_model = LogisticRegression(fit_intercept=True, penalty="l1", dual=False, C=1.0)
        y_values = np.ravel(y_values)

        rl_model.fit(x_values, y_values)

        return rl_model

    def random_forest_model(self):
        """No need to normalize data for this model."""

        labels = list(self.data.columns.values)

        y_values = self.data[["pred"]]

        l_x = list(filter(lambda x: x != 'pred', labels))

        x_values = self.data[l_x]

        rl_model = RandomForestClassifier(n_estimators=10, max_depth=5)
        y_values = np.ravel(y_values)

        lrfit = rl_model.fit(x_values, y_values)
        print('\nRandomForest score on full data set: {}\n'.format(lrfit.score(x_values, y_values)))
        ypred = rl_model.predict(x_values)
        print('\nConfusion matrix:')
        print(metrics.classification_report(y_values, ypred))

        return rl_model

    def normalize_dict_data(self, data):
        stdcols = ["restbp", "chol", "thalach", "oldpeak"]

        for k, v in data.items():
            data[k] = float(v)

        if self.model == LogisticRegression:
            print("TEST")
            for key in stdcols:
                data[key] = (data[key]-self.mean_dict[key])/self.std_dict[key]

            data["age"] = data["age"] / self.age_max
            data["ca"] = data["ca"] / self.ca_max

        return np.array([float(val) for val in data.values()]).reshape(1, -1)

    def predict(self, data):
        norm_data = self.normalize_dict_data(data)
        return self.model.predict(norm_data)


def main():
    heart_model = HeartModel()

    data_norm = heart_model.normalize_data(heart_model.data)

    labels = list(heart_model.data.columns.values)

    y_values = data_norm[["pred"]]
    l_x = list(filter(lambda x: x != 'pred', labels))

    print(heart_model.data)

    x_values = data_norm[l_x]

    rl_model = LogisticRegression(fit_intercept=True, penalty="l1", dual=False, C=1.0)

    y_values = np.ravel(y_values)

    lrfit = rl_model.fit(x_values, y_values)
    print('\nLogisticRegression score on full data set: {}\n'.format(lrfit.score(x_values, y_values)))
    ypred = rl_model.predict(x_values)
    print('\nConfusion matrix:')
    print(metrics.classification_report(y_values, ypred))


if __name__ == '__main__':
    main()
