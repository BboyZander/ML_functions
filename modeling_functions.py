import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import time
from collections import Counter, namedtuple

import os

import warnings

warnings.filterwarnings('ignore')

#  cv
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

# metrics
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

#
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

#
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression


# -------------------------------------------------------------------------------------------------


def validation(clf, X, y, n_folds=3, is_balance=True, random_state=123, verbose=1):
    '''
    Кросвалидация, тип балансировки данных - UnderSampling
    :param ensemble: модель
    :param X: Данные для кросвалидации
    :param y: флаги меток
    :param n_folds: количество фолдов
    :param balance: флаг балансировки
    :param random_state: random_state
    :verbose: количество выводимой информации
    :return: _
    '''
    rocs = []
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    model_name = str(clf)
    model_name = model_name[: model_name.find('(')]

    print('%s: ' % model_name)
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_index, :], X[test_index, :]
        y_train, y_test = y[train_index], y[test_index]

        if is_balance:
            rus = RandomUnderSampler(random_state=random_state)
            X_train, y_train = rus.fit_sample(X_train, y_train)

        clf.fit(X_train, y_train)
        try:
            roc_score = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
        except Exception:
            roc_score = roc_auc_score(y_test, clf.predict(X_test))

        if (i % verbose == 0) and (i not in [-1, 0]):
            print('%d fold: \n\tAUC: %.3f' % (i, roc_score))
        rocs.append(roc_score)

    print('\n--------\nAUCs: {}'.format(rocs))
    print('AUC: %.3f +/- %.3f' % (np.mean(rocs), np.std(rocs)), '\n--------')
    return


# CORRELATION
def delete_correlated_features(df, cut_off=0.75, exclude=[]):
    '''
    Функция, которая удаляет сильно скореллированые признаки
    :param df: исходный DataFrame
    :param cut_off: уровень корреляции
    :param exclude: список признаков, которые не будут удалены вне зависимости от силы корреляции
    :return: DataFrame без коррелированных признаков
    '''

    # Create correlation matrix
    corr_matrix = df.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Plotting All correlations
    if df.shape[1] <= 25:
        f, ax = plt.subplots(figsize=(15, 10))
        plt.title('All correlations', fontsize=20)
        sns.heatmap(df.corr(), annot=True)

    # Plotting highly correlated
    try:
        f, ax = plt.subplots(figsize=(15, 10))
        plt.title('High correlated', fontsize=20)
        sns.heatmap(corr_matrix[(corr_matrix > cut_off) & (corr_matrix != 1)].dropna(axis=0, how='all').dropna(axis=1,
                                                                                                               how='all'),
                    annot=True, linewidths=.5)
    except:
        print('No highly correlated features found')

    # Find index of feature columns with correlation greater than cut_off
    to_drop = [column for column in upper.columns if any(upper[column] > cut_off)]
    to_drop = [column for column in to_drop if column not in exclude]
    print('Dropped columns:', to_drop, '\n')
    df2 = df.drop(to_drop, axis=1)
    print('Features left after correlation check: {}'.format(len(df.columns) - len(to_drop)), '\n')

    print('Not dropped columns:', list(df2.columns), '\n')

    # Plotting final correlations
    if df2.shape[1] <= 25:
        f, ax = plt.subplots(figsize=(15, 10))
        plt.title('Final correlations', fontsize=20)
        sns.heatmap(df2.corr(), annot=True)
        plt.show()

    return df2


def quality_dynamic(*dfs, df_train, columns, id_column, cols_to_drop=[], target='def_flag', score_column='', legend=[]):
    """
    return DataFrame with delta GINI by dropping feature on each iteration
    :param dfs:
    :param df_train: dataframe for fitting
    :param columns: list of column names
    :param id_column: identificator column, inn for example
    :param score_column: name of column with score of model
    :param cols_to_drop:
    :param target: target column
    :param legend: name of your dataframes
    :return: df
    """
    assert len(dfs) == len(legend), 'len(dfs) != len(legend)'

    final_df = pd.DataFrame()
    # clf to get score of our model
    model = LogisticRegression(penalty='l2', C=0.1)

    # clf to get final score
    clf_sigmoid = LogisticRegression(penalty='l1', C=0.1)
    drop_sigmoid = [id_column, target, 'My_score']

    def inv_sigmoid(y):
        return np.log(y / (1 - y))

    for i, df in enumerate(dfs):
        base_GINI = (2 * roc_auc_score(df[target], df[score_column]) - 1) * 100
        f_add = []
        to_drop = []
        for feature in tqdm(columns[:0:-1]):
            if feature != '':
                to_drop.append(feature)
            scores = []
            temp_gini = []

            model.fit(df_train.drop(cols_to_drop + to_drop, axis=1), df_train[target].values)

            y_pred = model.predict_proba(df.drop(cols_to_drop + to_drop, axis=1))[:, 1]
            scores.append(y_pred)
            check_df = pd.DataFrame({'My_score': scores[0], 'def_flag': df[target],
                                     'id': df[id_column], 'score': df[score_column]})

            check_df['My_score_sigmoid'] = check_df['My_score'].apply(inv_sigmoid)

            clf_sigmoid.fit(check_df.drop(drop_sigmoid, axis=1), check_df[target])

            y_pred = clf_sigmoid.predict_proba(check_df.drop(drop_sigmoid, axis=1))[:, 1]
            temp_gini.append((2 * roc_auc_score(check_df[target], y_pred) - 1) * 100)

            f_add.append(temp_gini[0] - base_GINI)
        final_df[str(i)] = f_add

    if len(dfs) > 1:
        final_df['mean'] = final_df.mean(axis=1)
        legend = legend + ['mean']

    final_df.columns = legend
    final_df['Count of deleted features'] = [i for i in range(0, len(final_df))]

    return final_df
