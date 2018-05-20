import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC


class HeartModel:
    def __init__(self):
        self.data = HeartModel.read_data()
        self.dummies_for_categorical_data()

        all_labels = list(self.data.columns.values)
        y_values = self.data[["pred"]]
        l_x = list(filter(lambda x: x != 'pred', all_labels))
        self.labels = l_x
        self.x_values = self.data[l_x]
        self.y_values = np.ravel(y_values)

        # rl_model = RandomForestClassifier(random_state=0)
        # scores = self.randomize_search(rl_model)
        # params = scores.best_params_
        params = {'n_estimators': 1000, 'min_samples_split': 2, 'min_samples_leaf': 2,
                  'max_features': 3, 'max_depth': 5, 'bootstrap': False}
        self.model = self.random_forest_model(**params)

        print("Cross validation: {}".format(self.cross_val_model(self.model)))

        for feature in zip(self.labels, self.model.feature_importances_):
            print(feature)

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

        rl_model = RandomForestClassifier(random_state=0, **kwargs)
        lrfit = rl_model.fit(self.x_values, self.y_values)
        print('\nRandomForest score on full data set: {}\n'.format(lrfit.score(self.x_values, self.y_values)))
        ypred = rl_model.predict(self.x_values)
        print('\nConfusion matrix:')
        print(metrics.classification_report(self.y_values, ypred))

        return rl_model

    def cross_val_model(self, model):
        scores = cross_val_score(model, self.x_values, self.y_values, cv=4)
        return scores

    def randomize_search(self, model):
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start=5, stop=500, num=50)]
        # Number of features to consider at every split
        max_features = ['auto']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(2, 50, num=15)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 3, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4, 10]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}

        rf_random = RandomizedSearchCV(estimator=model, param_distributions=random_grid, n_iter=500, cv=3, verbose=0,
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
        data_2d = tsne.fit_transform(self.x_values)
        plt.figure(figsize=(6, 6))
        colors = ('r', 'g')
        for c, label in zip(colors, [0, 1]):
            indexes = [i for i in range(len(self.y_values)) if self.y_values[i] == label]
            plt.scatter(data_2d[indexes, 0], data_2d[indexes, 1], c=c, label=label)
        plt.legend()

        plt.title(plot_params)
        plt.show()

    def search_tsne_params(self):
        result_list = []
        perplexity = list(range(10, 160, 30))
        learning_rate = list(range(100, 500, 100))
        n_iter = list(range(1000, 10000, 2000))

        for p in perplexity:
            for lr in learning_rate:
                for i in n_iter:
                    params = {'perplexity': p, 'learning_rate': lr, 'n_iter': i}
                    tsne = TSNE(n_components=2, **params)
                    data_2d = tsne.fit_transform(self.x_values)
                    svm = LinearSVC(random_state=0)
                    svm.fit(data_2d, self.y_values)
                    score = svm.score(data_2d, self.y_values)
                    result_list.append((score, params))

        return result_list

    def normalize_dict_data(self, data):
        for k, v in data.items():
            data[k] = float(v)

        return np.array([float(val) for val in data.values()]).reshape(1, -1)

    def predict(self, data):
        norm_data = self.normalize_dict_data(data)
        return self.model.predict_proba(norm_data)[0][1]


def main():
    heart_model = HeartModel()
    rl_model = RandomForestClassifier(random_state=0)


    scores = heart_model.randomize_search(rl_model)
    a = scores.cv_results_['mean_test_score']
    index = sorted(range(len(a)), key=lambda i: a[i])[-5:][::-1]
    for i in index:
        print(scores.cv_results_['params'][i])

    # new_model = heart_model.random_forest_model(**scores.best_params_)
    new_model = RandomForestClassifier(random_state=0, **scores.best_params_)
    X_train, X_test, y_train, y_test = train_test_split(heart_model.x_values, heart_model.y_values, test_size=0.4,
                                                        random_state=0)
    new_model.fit(X_train, y_train)

    for feature in zip(heart_model.labels, new_model.feature_importances_):
        print(feature)
    sfm = SelectFromModel(new_model)
    sfm.fit(X_train, y_train)
    for feature_list_index in sfm.get_support(indices=True):
        print(heart_model.labels[feature_list_index])

    X_important_train = sfm.transform(X_train)
    X_important_test = sfm.transform(X_test)

    important_model = RandomForestClassifier(**scores.best_params_)
    important_model.fit(X_important_train, y_train)


    y_pred = new_model.predict(X_test)
    y_important_pred = important_model.predict(X_important_test)
    print("All features: {}".format(accuracy_score(y_test, y_pred)))
    print("Important features: {}".format(accuracy_score(y_test, y_important_pred)))


    #param_list = heart_model.search_tsne_params()
    #sort_params = sorted(param_list, key=lambda x: x[0])
    #print(sort_params[:3])
    # params = {'perplexity': 40, 'learning_rate': 200, 'n_iter': 7000}
    #heart_model.tsne_plot(**params)


if __name__ == '__main__':
    main()
