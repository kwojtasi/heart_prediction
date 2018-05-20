import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.model_selection import cross_val_score, RandomizedSearchCV


class HeartModel:
    def __init__(self):
        self.data = HeartModel.read_data()
        self.dummies_for_categorical_data()

        self.stdcols = ["restbp", "chol", "thalach", "oldpeak"]

        self.mean_dict = dict((key, self.data[key].mean()) for key in self.stdcols)
        self.std_dict = dict((key, self.data[key].std()) for key in self.stdcols)
        self.age_max = 100.0
        self.ca_max = 3.0

        labels = list(self.data.columns.values)
        y_values = self.data[["pred"]]
        l_x = list(filter(lambda x: x != 'pred', labels))
        self.x_values = self.data[l_x]
        self.y_values = np.ravel(y_values)

        rl_model = RandomForestClassifier()
        scores = self.param_search(rl_model)
        self.model = self.random_forest_model(**scores.best_params_)

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
    def normalize_data(data):
        data_norm = data.copy()
        stdcols = ["restbp", "chol", "thalach", "oldpeak"]
        data_norm[stdcols] = data_norm[stdcols].apply(lambda x: (x - x.mean()) / x.std())
        data_norm["age"] = data_norm["age"].apply(lambda x: x / 100.0)
        data_norm["ca"] = data_norm["ca"].apply(lambda x: x / 3.0)
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

    def random_forest_model(self, **kwargs):
        """No need to normalize data for this model."""

        rl_model = RandomForestClassifier(**kwargs)
        lrfit = rl_model.fit(self.x_values, self.y_values)
        print('\nRandomForest score on full data set: {}\n'.format(lrfit.score(self.x_values, self.y_values)))
        ypred = rl_model.predict(self.x_values)
        print('\nConfusion matrix:')
        print(metrics.classification_report(self.y_values, ypred))

        return rl_model

    def cross_val_model(self, model):
        scores = cross_val_score(model, self.x_values, self.y_values, cv=5)
        return scores

    def param_search(self, model):
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start=5, stop=50, num=5)]
        # Number of features to consider at every split
        max_features = ['auto']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(5, 20, num=5)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}

        print(random_grid)

        rf_random = RandomizedSearchCV(estimator=model, param_distributions=random_grid, n_iter=200, cv=3, verbose=2,
                                       random_state=42, n_jobs=-1)
        # Fit the random search model
        rf_random.fit(self.x_values, self.y_values)

        return rf_random

    def tsne_plot(self, **tsne_kwargs):
        print(tsne_kwargs)
        plot_params = "".join("({}={})".format(k, v) for k, v in tsne_kwargs.items())
        tsne = TSNE(n_components=2, **tsne_kwargs)
        data_2d = tsne.fit_transform(self.data)
        plt.figure(figsize=(6, 6))
        colors = ('r', 'g', 'b', 'c', 'm')
        for c, label in zip(colors, self.categories):
            indexes = [i for i in range(len(self.only_labels)) if self.only_labels[i] == label]
            plt.scatter(data_2d[indexes, 0], data_2d[indexes, 1], c=c, label=label)
        plt.legend()

        plt.title(plot_params)
        plt.show()

    def normalize_dict_data(self, data):
        stdcols = ["restbp", "chol", "thalach", "oldpeak"]

        for k, v in data.items():
            data[k] = float(v)

        if self.model == LogisticRegression:
            print("TEST")
            for key in stdcols:
                data[key] = (data[key] - self.mean_dict[key]) / self.std_dict[key]

            data["age"] = data["age"] / self.age_max
            data["ca"] = data["ca"] / self.ca_max

        return np.array([float(val) for val in data.values()]).reshape(1, -1)

    def predict(self, data):
        norm_data = self.normalize_dict_data(data)
        print(norm_data)
        print(self.model.predict(norm_data))
        return self.model.predict_proba(norm_data)[0][1]


def main():
    # bow.tsne_plot(random_state=1, perplexity=40, learning_rate=50)
    heart_model = HeartModel()
    rl_model = RandomForestClassifier()
    scores = heart_model.param_search(rl_model)
    print(scores.best_params_)

    new_model = heart_model.random_forest_model(**scores.best_params_)



def main2():
    # bow.tsne_plot(random_state=1, perplexity=40, learning_rate=50)
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
