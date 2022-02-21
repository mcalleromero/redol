# -*- coding: utf-8 -*-
#!/usr/bin/python3
from __future__ import division

from collections import defaultdict
from random import shuffle

import numpy as np
import pymp
from sklearn import tree
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder


class RedolClassifierException(Exception):
    pass


class RedolClassifier:
    def __init__(
        self,
        n_estimators=100,
        method="distributed",
        pil=0.75,
        bootstrap=1.0,
        nearest_neighbours=None,
        n_jobs=8,
        max_depth=None,
    ):
        self.n_estimators = n_estimators
        self.method = method
        self.pil = pil
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.max_depth = max_depth
        self.nearest_neighbours = nearest_neighbours

    def get_params(self, deep=True):
        # TODO: With deep true it should return base learners params too
        return {
            "n_estimators": self.n_estimators,
            "pil": self.pil,
            "bootstrap": self.bootstrap,
            "n_jobs": self.n_jobs,
            "max_depth": self.max_depth,
            "method": self.method,
            "nearest_neighbours": self.nearest_neighbours,
        }

    def set_params(self, **params):
        valid_params = self.get_params(deep=True)
        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if key not in valid_params:
                raise ValueError(
                    "Invalid parameter %s for estimator %s. "
                    "Check the list of available parameters "
                    "with `estimator.get_params().keys()`." % (key, self)
                )

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

    def fit(self, x, y):
        """
        This method is used to fit each one of the decision trees the random noise classifier is composed with.
        This is the way to fit the complete classifier and it is compulsory to carry on with the data classification.

        :param x: original features from the dataset
        :param y: original classes from the dataset
        """
        self.classes = np.unique(y)

        self.enc = OneHotEncoder(handle_unknown="ignore")
        self.enc.fit(y.reshape(-1, 1))

        # Nearest Neighbors returns the k nearest neighbors for a X example,
        # including itself, so it is recommended using an even number of k.
        # For example, if 4 is used as k, 3 different nn will be used excluding
        # the instance itself.
        if self.nearest_neighbours:
            nn = NearestNeighbors()
            nn.fit(x)
            self.nn_indixes = nn.kneighbors(
                X=x, n_neighbors=self.nearest_neighbours, return_distance=False
            )
            self.original_y = y

        self.classifiers = pymp.shared.list()

        with pymp.Parallel(self.n_jobs) as p:
            for n_classifier in p.range(0, self.n_estimators):
                clf = tree.DecisionTreeClassifier(max_depth=self.max_depth)

                # Bootstrap extractions to train the model.
                # These extractions can repeat some indices.
                number_of_extractions = int(self.bootstrap * x.shape[0])
                ind = np.random.randint(0, x.shape[0], number_of_extractions)
                _x = x[ind, :]
                _y = y[ind]

                # If nearest neighbours are set, the method parameter is ignored
                # as it will be using the k nearest neighbours
                if self.nearest_neighbours and self.method == "no_pil":
                    try:
                        (
                            modified_x,
                            modified_y,
                        ) = self._change_class_nearest_neighbours_no_pil(
                            _x, _y, indexes=ind
                        )
                    except TypeError as e:
                        err_msg = f"Nearest neighbours parameter must be None or an integer within (1, N), N being X instances. It was set to {self.nearest_neighbours}"
                        raise RedolClassifierException(err_msg)
                if self.nearest_neighbours:
                    try:
                        modified_x, modified_y = self._change_class_nearest_neighbours(
                            _x, _y, indexes=ind
                        )
                    except TypeError as e:
                        err_msg = f"Nearest neighbours parameter must be None or an integer within (1, N), N being X instances. It was set to {self.nearest_neighbours}"
                        raise RedolClassifierException(err_msg)
                elif self.method == "regular":
                    modified_x, modified_y = self._change_class(_x, _y)
                elif self.method == "distributed":
                    modified_x, modified_y = self._change_class_distributed(_x, _y)
                elif self.method == "randomized":
                    modified_x, modified_y = self._change_class_randomized(_x, _y)
                else:
                    err_msg = f"The method {self.method} is not a valid method: regular, distributed, randomized, no_pil (with nearest_neighbors)"
                    raise RedolClassifierException(err_msg)

                clf.fit(modified_x, modified_y)

                # self.classifiers[n_classifier] = clf
                with p.lock:
                    self.classifiers.append(clf)

    def score(self, x, y):
        """
        This method is used to calculate the classifier accuracy comparing the obtained classes with the original
        ones from the dataset.

        :param x: original features from the dataset
        :param y: original classes from the dataset
        :param suggested_class: a new feature added to classify the examples
        :return: classifier accuracy
        """
        return (
            sum(
                [
                    1
                    for i, prediction in enumerate(self.predict(x))
                    if prediction == y[i]
                ]
            )
            / x.shape[0]
        )

    def predict(self, x):
        """
        This method is used to generate the class predictions from each example to be classified.
        It uses the method predict_proba to calculate the probabilities that a data is well classified or not.

        :param x: original features from the dataset
        :param suggested_class: a new feature added to classify the examples
        :return: an array with the predicted class for each example from the dataset
        """
        return np.array(
            [
                float(np.where(pred == np.amax(pred))[0][0])
                for pred in self.predict_proba(x)
            ]
        )

    def predict_proba(self, x):
        """
        This method calculates the probability that a data is well classified or not. It adds a new feature
        to the dataset depending on the suggested_class attribute.

        :param x: data to be classified
        :param suggested_class: new feature to be added
        :return: probabilities that a data is well classified or not
        """
        predictions = []

        for cl in self.classes:
            preds = []

            _x = x.copy()
            if len(self.classes) > 2:
                _x = np.c_[
                    _x,
                    np.tile(
                        self.enc.transform(cl.reshape(-1, 1)).toarray(), (x.shape[0], 1)
                    ),
                ]
            else:
                _x = np.c_[_x, np.repeat(cl, x.shape[0])]

            [preds.append(clf.predict_proba(_x)) for clf in self.classifiers]
            preds = np.array(preds).mean(axis=0)
            predictions.append(preds[:, 1])

        return np.array(predictions).transpose()

    def predict_proba_error(self, x):
        """
        This method calculates a matrix which contains the probabilities of each example cumulatively.

        :param x: the original features from the dataset
        :param suggested_class: the class the classifier uses as new feature
        :return: the final probabilities matrix
        """
        self.predictions = []

        for cl in self.classes:
            preds = []

            _x = x.copy()
            if len(self.classes) > 2:
                _x = np.c_[
                    _x,
                    np.tile(
                        self.enc.transform(cl.reshape(-1, 1)).toarray(), (x.shape[0], 1)
                    ),
                ]
            else:
                _x = np.c_[_x, np.repeat(cl, x.shape[0])]

            [preds.append(clf.predict_proba(_x)) for clf in self.classifiers]
            preds = np.array(preds)

            for i in range(len(self.classifiers) - 1, -1, -1):
                preds[i, :, :] = preds[: i + 1, :, :].sum(axis=0)
                preds[i, :, :] /= i + 1

            self.predictions.append(preds[:, :, 1].transpose())

        self.predictions = np.array(self.predictions).transpose()

        return self.predictions

    def score_error(self, x, y, n_classifiers=100):
        """
        With this method we are able to see what is going on with the classification of the examples for each classifier.
        This method allows us to calculate the score obtained using the amount of classifiers we want up to the maximum
        of classifiers with which it was declared.

        :param x: original features dataset
        :param y: original classes from the dataset
        :param n_classifiers: number of classifiers used to calculate the score
        :return: score obtained
        """
        if n_classifiers is None:
            n_classifiers = len(self.classifiers)

        n_classifiers -= 1

        return (
            sum(
                [
                    1
                    for i, pred in enumerate(self.predictions[n_classifiers, :, :])
                    if float(np.where(pred == np.amax(pred))[0][0] == y[i])
                ]
            )
            / x.shape[0]
        )

    ###################################################################################################################
    ###################################################################################################################
    ###################################################################################################################

    def _change_class(self, x, y):
        """
        Given a data set split in features and classes this method transforms this set into another set.
        This new set is created based on random noise generation and its classification. The randomization
        of the new data set is given by the percentage received from the class constructor.

        The randomization is generated changing some data classes and pointing it out in the new class.
        The new class is calculated comparing the original class with the data randomization. For this new class
        '1' means "well classified" and '0', the opposite.

        :param x: features from the original data set
        :param y: classes from the original data set
        :return: features and classes from the new data set
        """
        data = np.c_[x, y]

        num_data = data.shape[0]

        percentage = int(num_data * self.pil)

        updated_data = data.copy()

        random_data = list(range(0, num_data))
        shuffle(random_data)

        if len(self.classes) <= 2:
            updated_data = self._binary(percentage, random_data, updated_data)
        else:
            updated_data = self._multiclass(percentage, random_data, updated_data, y)

        updated_class = [
            (updated_data[i, -1] == data[i, -1]) for i in range(0, num_data)
        ]

        # Changes the old class from the data features to the one hot encoder ones
        if len(self.classes) > 2:
            updated_data = np.c_[
                updated_data[:, :-1],
                self.enc.transform(updated_data[:, -1].reshape(-1, 1)).toarray(),
            ]

        return updated_data, np.array(updated_class)

    def _change_class_distributed(self, x, y):
        data = np.c_[x, y]

        y = y.astype("int64")

        # With this formulae: (bin_count[ii]**2).sum()
        # we are doing the sum of all the class distributions
        # power 2
        bin_count = np.bincount(y)
        ii = np.nonzero(bin_count)[0]

        # w is a formula described in Breiman, Randomizin Outputs
        # and it means the proportion of class k labels flipped is
        # w * (1 - k class distribution). The total proportion of
        # flipped instances is w * (k class distribution) * (1 - k class distribution)
        w = self.pil / (1 - ((bin_count[ii] / y.shape) ** 2).sum())

        final_data = []
        final_class = []

        for c_dist in zip(ii, bin_count[ii] / y.shape):
            updated_data = data[data[:, -1] == c_dist[0], :].copy()
            original_data = updated_data.copy()
            num_data = updated_data.shape[0]
            percentage = int((c_dist[1] * w) * num_data)

            random_data = list(range(0, num_data))
            shuffle(random_data)

            if len(self.classes) <= 2:
                updated_data = self._binary(percentage, random_data, updated_data)
            else:
                updated_data = self._multiclass(
                    percentage, random_data, updated_data, y
                )

            updated_class = [
                (updated_data[i, -1] == original_data[i, -1])
                for i in range(0, num_data)
            ]

            final_data.append(np.array(updated_data))
            final_class.append(np.array(updated_class))

        final_data = np.concatenate(final_data, axis=0)
        final_class = np.concatenate(final_class, axis=0)

        # Changes the old class from the data features to the one hot encoder ones
        if len(self.classes) > 2:
            final_data = np.c_[
                final_data[:, :-1],
                self.enc.transform(final_data[:, -1].reshape(-1, 1)).toarray(),
            ]

        return final_data, final_class

    def _change_class_randomized(self, x, y):
        data = np.c_[x, y]

        y = y.astype("int64")

        num_data = data.shape[0]
        updated_data = data.copy()

        # The attr y' corresponds to the original classes
        # but shuffled
        y_new = y.copy()
        shuffle(y_new)
        updated_data[:, -1] = y_new
        updated_class = [
            (updated_data[i, -1] == data[i, -1]) for i in range(0, num_data)
        ]

        # Changes the old class from the data features to the one hot encoder ones
        if len(self.classes) > 2:
            updated_data = np.c_[
                updated_data[:, :-1],
                self.enc.transform(updated_data[:, -1].reshape(-1, 1)).toarray(),
            ]

        return updated_data, np.array(updated_class)

    def _change_class_nearest_neighbours(self, x, y, indexes):
        """
        Given a data set split in features and classes this method transforms this set into another set.
        This new set is created based on random noise generation and its classification. The randomization
        of the new data set is given by the percentage received from the class constructor.

        The randomization is generated changing some data classes and pointing it out in the new class.
        The new class is calculated comparing the original class with the data randomization. For this new class
        '1' means "well classified" and '0', the opposite.

        :param x: features from the original data set
        :param y: classes from the original data set
        :return: features and classes from the new data set
        """
        data = np.c_[x, y]

        num_data = data.shape[0]

        percentage = int(num_data * self.pil)

        updated_data = data.copy()

        random_data = list(range(0, num_data))
        shuffle(random_data)

        updated_data = self._nearest_neighbours(
            percentage, random_data, updated_data, y, indexes
        )

        updated_class = [
            (updated_data[i, -1] == data[i, -1]) for i in range(0, num_data)
        ]

        # Changes the old class from the data features to the one hot encoder ones
        if len(self.classes) > 2:
            updated_data = np.c_[
                updated_data[:, :-1],
                self.enc.transform(updated_data[:, -1].reshape(-1, 1)).toarray(),
            ]

        return updated_data, np.array(updated_class)

    def _change_class_nearest_neighbours_no_pil(self, x, y, indexes):
        """
        Given a data set split in features and classes this method transforms this set into another set.
        This new set is created based on random noise generation and its classification. The randomization
        of the new data set is given by the percentage received from the class constructor.

        The randomization is generated changing some data classes and pointing it out in the new class.
        The new class is calculated comparing the original class with the data randomization. For this new class
        '1' means "well classified" and '0', the opposite.

        :param x: features from the original data set
        :param y: classes from the original data set
        :return: features and classes from the new data set
        """
        data = np.c_[x, y]
        num_data = data.shape[0]
        updated_data = data.copy()

        updated_data = self._nearest_neighbours_no_pil(updated_data, y, indexes)

        updated_class = [
            (updated_data[i, -1] == data[i, -1]) for i in range(0, num_data)
        ]

        # Changes the old class from the data features to the one hot encoder ones
        if len(self.classes) > 2:
            updated_data = np.c_[
                updated_data[:, :-1],
                self.enc.transform(updated_data[:, -1].reshape(-1, 1)).toarray(),
            ]

        return updated_data, np.array(updated_class)

    ###################################################################################################################
    ###################################################################################################################
    ###################################################################################################################

    def _binary(self, percentage, random_data, updated_data):
        for num in random_data[:percentage]:
            updated_data[num, -1] = 1 - updated_data[num, -1]

        return updated_data

    def _multiclass(self, percentage, random_data, updated_data, y):
        classes = list(set(y))

        for num in random_data[:percentage]:
            prev_class = updated_data[num, -1]
            classes_without_prev_class = classes.copy()  # copy classes list
            classes_without_prev_class.remove(prev_class)
            updated_data[num, -1] = choice(classes_without_prev_class)

        return updated_data

    def _nearest_neighbours_no_pil(self, updated_data, y, indexes):
        classes = set(y)

        for num in range(updated_data.shape[0]):
            prev_class = updated_data[num, -1]
            # This are the nearest neighbors indexes. We can get their original classes too.
            nn_classes_idx = self.nn_indixes[indexes[num]]
            classes_without_prev_class = self.original_y[nn_classes_idx]
            classes_without_prev_class = set(classes_without_prev_class)
            classes_without_prev_class.discard(prev_class)
            # If classes from neighbours are equal, class is not changed
            if not classes_without_prev_class:
                continue

            updated_data[num, -1] = choice(list(classes_without_prev_class))

        return updated_data

    def _nearest_neighbours(self, percentage, random_data, updated_data, y, indexes):
        classes = set(y)

        for num in random_data[:percentage]:
            prev_class = updated_data[num, -1]
            # This are the nearest neighbors indexes. We can get their original classes too.
            nn_classes_idx = self.nn_indixes[indexes[num]]
            classes_without_prev_class = self.original_y[nn_classes_idx]
            classes_without_prev_class = set(classes_without_prev_class)
            classes_without_prev_class.discard(prev_class)
            # If classes from neighbours are equal, we choose randomly from any other class
            if not classes_without_prev_class:
                classes_without_prev_class = classes.copy()
                classes_without_prev_class.discard(prev_class)

            updated_data[num, -1] = choice(list(classes_without_prev_class))

        return updated_data
