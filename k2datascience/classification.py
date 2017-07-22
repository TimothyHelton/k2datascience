#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Classification Module

.. moduleauthor:: Timothy Helton <timothy.j.helton@gmail.com>
"""
from collections import Counter
import logging
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, log_loss
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import statsmodels.api as sm
import statsmodels.formula.api as smf

from k2datascience.utils import ax_formatter, size, save_fig


log_format = ('%(asctime)s  %(levelname)8s  -> %(name)s <- '
              '(line: %(lineno)d) %(message)s\n')
date_format = '%m/%d/%Y %I:%M:%S'
logging.basicConfig(format=log_format, datefmt=date_format,
                    level=logging.INFO)

current_dir = osp.dirname(osp.realpath(__file__))
data_dir = osp.realpath(osp.join(current_dir, '..', 'data', 'classification'))
auto_data = osp.join(data_dir, 'auto.csv')
boston_data = osp.join(data_dir, 'boston.csv')
weekly_data = osp.join(data_dir, 'weekly.csv')


class Classify:
    """
    Base class for classification.

    :Attributes:

    - **classification** *str* classification report
    - **confusion** *DataFrame* confusion matrix
    - **data**: *DataFrame* data
    - **log_loss**: *float* cross-entropy loss
    - **model**: classification model type
    - **predict**: *ndarray* model predicted values
    - **x_train**: *DataFrame* training features
    - **y_train**: *Series* training response
    - **x_test**: *DataFrame* testing features
    - **y_test**: *Series* testing response
    """
    def __init__(self):
        self.classification = None
        self.confusion = None
        self.data = None
        self._log_loss = None
        self.model = None
        self.predict = None
        self._score = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def __repr__(self):
        return 'Classify()'

    @property
    def log_loss(self):
        return f'Log Loss: {self._log_loss:.3f}'

    @property
    def score(self):
        return f'Model Score: {self._score:.3f}'

    def accuracy_vs_k(self, max_k=20, save=False):
        """
        Print the accuracy results for multiple values of K for a KNN model.

        :param int max_k: largest K value to be tested
        :param bool save: if True the figure will be saved
        """
        accuracy = {}
        for n in range(1, max_k, 1):
            self.classify_data(model='KNN', n=n)
            accuracy[n] = self._score

        fig = plt.figure('KNN Accuracy vs K', figsize=(8, 6),
                         facecolor='white', edgecolor='black')
        rows, cols = (1, 1)
        ax0 = plt.subplot2grid((rows, cols), (0, 0))

        result = pd.Series(accuracy)
        result.plot(ax=ax0)

        ax0.set_title('Accuracy vs Nearest Neighbors Quantity',
                      fontsize=size['title'])
        ax0.set_xlabel('Nearest Neighbors Quantity $K$',
                       fontsize=size['label'])
        ax0.set_ylabel('Accuracy', fontsize=size['label'])
        ax0.yaxis.set_major_formatter(ax_formatter['percent'])

        save_fig('accuracy_vs_k', save)

    def classify_data(self, model='LR', n=1):
        """
        Classify Data

        :param str model: model designator (see table below for \
            implemented types)
        :param int n: number of nearest neighbors to evaluate

        +------------------+-------------------------------+
        | Model Designator | Scikit-Learn Model Type       |
        +==================+===============================+
        | KNN              | KNeighborsClassifier          |
        +------------------+-------------------------------+
        | LDA              | LinearDiscriminantAnalysis    |
        +------------------+-------------------------------+
        | LR               | LogisticRegression            |
        +------------------+-------------------------------+
        | NB               | GaussianNB                    |
        +------------------+-------------------------------+
        | QDA              | QuadraticDiscriminantAnalysis |
        +------------------+-------------------------------+
        """
        models = {
            'KNN': sklearn.neighbors.KNeighborsClassifier(n_neighbors=n),
            'LDA': LinearDiscriminantAnalysis(),
            'LR': LogisticRegression(),
            'NB': GaussianNB(),
            'QDA': QuadraticDiscriminantAnalysis(),
        }

        if model not in models.keys():
            logging.error(f'Requested model {model} has not been implemented.')

        self.model = (models[model]
                      .fit(self.x_train, self.y_train))
        self.predict = self.model.predict(self.x_test)
        self.confusion = pd.DataFrame(confusion_matrix(self.y_test,
                                                       self.predict))
        self.classification = classification_report(self.y_test,
                                                    self.predict)
        self._log_loss = log_loss(self.y_test,
                                  self.model.predict_proba(self.x_test))

        self._score = self.model.score(self.x_test, self.y_test)


class Auto(Classify):
    """
    Attributes and methods related to the auto dataset.

    :Attributes:

    - **classification** *str* classification report
    - **confusion** *DataFrame* confusion matrix
    - **data**: *DataFrame* data
    - **data_file**: *str* path to data file
    - **data_types**: *dict* data type definitions
    - **model**: classification model type
    - **predict**: *ndarray* model predicted values
    - **x_train**: *DataFrame* training features
    - **y_train**: *Series* training response
    - **x_test**: *DataFrame* testing features
    - **y_test**: *Series* testing response
    """
    def __init__(self):
        super().__init__()
        self.classification = None
        self.confusion = None
        self.data = None
        self.data_file = auto_data
        self.data_types = {
            'mpg': np.float64,
            'cylinders': np.int32,
            'displacement': np.float64,
            'horsepower': np.int32,
            'weight': np.int32,
            'acceleration': np.float64,
            'year': np.int32,
            'origin': np.int32,
            'name': str,
        }
        self.model = None
        self.predict = None

        self.train_pct = 0.8
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

        self.load_data()

    def __repr__(self):
        return 'Auto()'

    def box_plots(self, save=False):
        """
        Box plot of MPG vs Cylinders and MPG vs Origin

        :param bool save: if True the figure will be saved
        """
        fig = plt.figure('Correlation Heatmap', figsize=(12, 5),
                         facecolor='white', edgecolor='black')
        rows, cols = (1, 2)
        ax0 = plt.subplot2grid((rows, cols), (0, 0))
        ax1 = plt.subplot2grid((rows, cols), (0, 1), sharey=ax0)

        sns.boxplot(x='cylinders', y='mpg', data=self.data, width=0.4, ax=ax0)

        ax0.set_title('MPG vs Cylinders', fontsize=size['title'])
        ax0.set_xlabel('Cylinders', fontsize=size['label'])

        sns.boxplot(x='origin', y='mpg', data=self.data, width=0.4, ax=ax1)

        ax1.set_title('MPG vs Origin', fontsize=size['title'])
        ax1.set_xlabel('Origin', fontsize=size['label'])

        for ax in (ax0, ax1):
            ax.set_ylabel('MPG', fontsize=size['label'])

        plt.suptitle('Auto Dataset', fontsize=size['super_title'], y=1.03)

        save_fig('mpg_vs_cylinders', save)

    def load_data(self):
        """
        Load the data into a DataFrame
        """
        def date_parse(year):
            """
            Convert year from YY to 19YY
            :param str year: year to be converted
            :return: year in 19YY format
            :rtype: datetime
            """
            return pd.datetime.strptime(f'19{year}', '%Y')
        self.data = (pd.read_csv(self.data_file,
                                 dtype=self.data_types,
                                 header=None,
                                 index_col=6,
                                 names=self.data_types.keys(),
                                 parse_dates=[6],
                                 date_parser=date_parse,
                                 skiprows=1,
                                 ))
        binary_mpg = self.data.mpg.values.copy()
        binary_mpg_mean = binary_mpg.mean()
        binary_mpg[self.data.mpg < binary_mpg_mean] = 0
        binary_mpg[self.data.mpg >= binary_mpg_mean] = 1

        features = ['displacement', 'horsepower', 'weight', 'origin']

        train_idx = int(self.data.shape[0] * self.train_pct)
        self.x_train = (self.data
                        .loc[:, features]
                        .select_dtypes(exclude=['object'])[:train_idx])
        self.y_train = pd.Series(binary_mpg[:train_idx])
        self.x_test = (self.data
                       .loc[:, features]
                       .select_dtypes(exclude=['object'])[train_idx:])
        self.y_test = pd.Series(binary_mpg[train_idx:])

        self.data['binary_mpg'] = binary_mpg


class WildFaces(Classify):
    """
    Attributes and methods related to the Labeled Faces in the Wild Dataset.

    :Attributes:

    - **data**: *ndarray* data containing the features
    - **images**: *ndarray* pixel information for each image
    - **pca**: *sklearn.PCA* scikit-learn instance of PCA class
    - **target**: *ndarray* numerical values for each target
    - **target_names**: *ndarray* categorical value for each target
    - **var_pct**: *pd.Series* principle components variance percentage
    - **var_pct_cum**: *pd.Series* principle components cumulative variance /
        percentage
    """
    def __init__(self, n_faces=70):
        super().__init__()
        self.dataset = fetch_lfw_people(min_faces_per_person=n_faces,
                                        resize=0.4)
        self.data = self.dataset['data']
        self.images = self.dataset['images']
        self.pca = None
        self.target = self.dataset['target']
        self.target_names = self.dataset['target_names']
        self.var_pct = None
        self.x_train, self.x_test, self.y_train, self.y_test = (
            train_test_split(self.data, self.target, random_state=0)
        )

    def avg_face_plot(self):
        """
        Plot the average face determined by PCA.
        """
        if self.pca is None:
            self.calc_pca()

        fig = plt.figure(figsize=(8, 6))
        plt.imshow(self.pca.mean_.reshape(self.images[0].shape),
                   cmap=plt.cm.bone)

    def calc_pca(self, n_components=150):
        """
        Calculate the data's Principal Components.

        :param int n_components: number of principal components to return
        """
        self.pca = PCA(n_components=n_components)
        self.pca.fit(self.x_train)

        self.var_pct = pd.Series(self.pca.explained_variance_ratio_)

    def components_plot(self, n=30):
        """
        Plot to the principal components.

        :param int n: number of components to plot
        """
        fig = plt.figure(figsize=(16, 6))
        for comp in range(n):
            ax = fig.add_subplot(3, 10, comp + 1, xticks=[], yticks=[])
            ax.imshow(self.pca.components_[comp].reshape((50, 37)),
                      cmap=plt.cm.bone)

    def faces_plot(self, n=15):
        """
        Plot the first n faces of the dataset.

        :param int n: number of faces to plot
        """
        fig = plt.figure(figsize=(8, 6))
        for im in range(n):
            ax = fig.add_subplot(3, 5, im + 1, xticks=[], yticks=[])
            ax.imshow(self.images[im], cmap=plt.cm.bone)

    def targets_barplot(self, save=False):
        """
        Create bar plot of targets.

        :param bool save: if True the figure will be saved
        """
        fig = plt.figure('Correlation Heatmap', figsize=(12, 5),
                         facecolor='white', edgecolor='black')
        rows, cols = (1, 2)
        ax0 = plt.subplot2grid((rows, cols), (0, 0))

        counts = Counter(self.target)
        (pd.Series({self.target_names[x]: counts[x]
                    for x in range(self.target_names.size)})
         .plot(kind='bar', alpha=0.5, edgecolor='black', ax=ax0))

        ax0.set_title('Labeled Faces in the Wild', fontsize=size['title'])
        ax0.set_xlabel('Target', fontsize=size['label'])
        ax0.set_xticklabels(ax0.xaxis.get_majorticklabels(), rotation=80)
        ax0.set_ylabel('Count', fontsize=size['label'])

        save_fig('lfw_target_boxplot', save)


class Weekly:
    """
    Attributes and methods related to the weekly dataset.

    :Attributes:

    - **classification** *str* classification report
    - **confusion** *DataFrame* confusion matrix
    - **data**: *DataFrame* data
    - **data_file**: *str* path to data file
    - **data_types**: *dict* data type definitions
    - **knn_model**: *KNeighborsClassifier* K-Nearest Neighbors model
    - **lda_model**: *LinearDiscriminantAnalysis* LDA model
    - **logistic_formula**: *str* logistic regression formula
    - **logistic_model**: *GLMResultsWrapper* statsmodels logistic regression \
        model
    - **predict**: *ndarray* model predicted values
    - **predicted_prob**: *ndarray* cutoff probability to make classification
    - **prediction_nom**: *ndarray* binary normalized predicted values
    - **qda_model**: *QuadraticDiscriminantAnalysis* QDA model
    - **train_pct**: *float* percentage of data to be used for training
    - **x_train**: *DataFrame* training features
    - **y_train**: *Series* training response
    - **x_test**: *DataFrame* testing features
    - **y_test**: *Series* testing response
    """
    def __init__(self):
        self.classification = None
        self.confusion = None
        self.data = None
        self.data_file = weekly_data
        self.data_types = {
            'idx': np.int32,
            'year': np.int64,
            'lag1': np.float64,
            'lag2': np.float64,
            'lag3': np.float64,
            'lag4': np.float64,
            'lag5': np.float64,
            'volume': np.float64,
            'today': np.float64,
            'direction': str,
        }

        self.knn_model = None

        self.lda_model = None

        lags = ' + '.join([f'lag{x}' for x in range(1, 6, 1)])
        self.logistic_formula = f'direction ~ {lags} + volume'
        self.logistic_model = None
        self.predict = None
        self.predicted_prob = 0.5
        self.prediction_nom = None

        self.qda_model = None

        self.train_pct = 0.8
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

        self.load_data()

    def __repr__(self):
        return 'Weekly()'

    def calc_prediction(self, actual, predicted):
        """
        Predict response of logistic_model based on given values.

        :param Series actual: actual response values
        :param Series predicted: predicted response values
        """
        self.confusion = pd.DataFrame(confusion_matrix(actual, predicted))
        self.classification = classification_report(actual, predicted)

    def categorize(self, data):
        """
        Convert predicted values into categories

        :param DataFrame data: data to be used to categorize predictions
        """
        self.predict = self.logistic_model.predict(data)
        self.prediction_nom = self.predict.copy()
        self.prediction_nom[self.prediction_nom >= self.predicted_prob] = 1
        self.prediction_nom[self.prediction_nom < self.predicted_prob] = 0

    def knn(self, n=1):
        """
        K-Nearest Neighbors Analysis of the data.

        :param int n: number of nearest neighbors to consider
        """
        self.knn_model = (sklearn.neighbors
                          .KNeighborsClassifier(n_neighbors=n)
                          .fit(self.x_train.drop('direction', axis=1),
                               self.y_train))
        self.predict = (self.knn_model
                        .predict(self.x_test.drop('direction', axis=1)))
        self.calc_prediction(actual=self.y_test, predicted=self.predict)

    def load_data(self):
        """
        Load the data into a DataFrame
        """
        self.data = (pd.read_csv(self.data_file,
                                 dtype=self.data_types,
                                 header=None,
                                 index_col=1,
                                 names=self.data_types.keys(),
                                 parse_dates=[1],
                                 skiprows=1,
                                 )
                     .drop('idx', axis=1))
        self.data.direction = (self.data.direction
                               .astype('category'))

        train_idx = int(self.data.shape[0] * self.train_pct)
        self.x_train = self.data[:train_idx]
        self.y_train = self.data.direction[:train_idx].cat.codes
        self.x_test = self.data[train_idx:]
        self.y_test = self.data.direction[train_idx:].cat.codes

    def lda(self):
        """
        Linear Discriminate Analysis of the data.
        """
        self.lda_model = (LinearDiscriminantAnalysis()
                          .fit(self.x_train.drop('direction', axis=1),
                               self.y_train))
        self.predict = (self.lda_model
                        .predict(self.x_test.drop('direction', axis=1)))
        self.calc_prediction(actual=self.y_test, predicted=self.predict)

    def logistic_regression(self, data):
        """
        Create logistic regression model with direction as the result.

        :param DataFrame data: data features including response
        """
        self.logistic_model = (smf.glm(formula=self.logistic_formula,
                                       data=data,
                                       family=sm.families.Binomial())
                               .fit())

        self.categorize(data)
        y_test = data.direction.cat.codes
        self.calc_prediction(actual=y_test, predicted=self.prediction_nom)

    def qda(self):
        """
        Quadratic Discriminate Analysis of the data.
        """
        self.qda_model = (QuadraticDiscriminantAnalysis()
                          .fit(self.x_train.drop('direction', axis=1),
                               self.y_train))
        self.predict = (self.qda_model
                        .predict(self.x_test.drop('direction', axis=1)))
        self.calc_prediction(actual=self.y_test, predicted=self.predict)
