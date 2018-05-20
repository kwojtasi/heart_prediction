import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, GridSearchCV


class HeartModel:
    def __init__(self):
        self.data = HeartModel.read_data()
        self.dummies_for_categorical_data()

        labels = list(self.data.columns.values)
        y_values = self.data[["pred"]]
        l_x = list(filter(lambda x: x != 'pred', labels))
        self.x_values = self.data[l_x]
        self.y_values = np.ravel(y_values)

        rl_model = RandomForestClassifier()
        scores = self.randomize_search(rl_model)
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

    def randomize_search(self, model):
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start=5, stop=50, num=20)]
        # Number of features to consider at every split
        max_features = ['auto']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(2, 20, num=10)]
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

        rf_random = RandomizedSearchCV(estimator=model, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                                       random_state=42, n_jobs=-1)
        # Fit the random search model
        rf_random.fit(self.x_values, self.y_values)

        return rf_random

    def grid_search(self, model):
        # Number of trees in random forest
        n_estimators = list(range(10, 30, 1))
        # Number of features to consider at every split
        max_features = ['auto']
        # Maximum number of levels in tree
        max_depth = list(range(2, 10, 2))
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 3, 4]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True]
        # Create the random grid
        param_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}

        rf_random = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose=2, n_jobs=-1)
        # Fit the random search model
        rf_random.fit(self.x_values, self.y_values)

        return rf_random

    def tsne_plot(self, **tsne_kwargs):
        print(tsne_kwargs)
        plot_params = "".join("({}={})".format(k, v) for k, v in tsne_kwargs.items())
        tsne = TSNE(n_components=2, **tsne_kwargs)
        data_2d = tsne.fit_transform(self.data)
        plt.figure(figsize=(6, 6))
        colors = ('r', 'g')
        for c, label in zip(colors, [0, 1]):
            indexes = [i for i in range(len(self.y_values)) if self.y_values[i] == label]
            plt.scatter(data_2d[indexes, 0], data_2d[indexes, 1], c=c, label=label)
        plt.legend()

        plt.title(plot_params)
        plt.show()

    def normalize_dict_data(self, data):
        for k, v in data.items():
            data[k] = float(v)

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
    scores = heart_model.randomize_search(rl_model)

    a = scores.cv_results_['mean_test_score']
    new_model = heart_model.random_forest_model(**scores.best_params_)

    # heart_model.tsne_plot()


if __name__ == '__main__':
    main()
