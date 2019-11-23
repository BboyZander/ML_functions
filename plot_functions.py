import matplotlib as plt
import numpy as np
from sklearn.metrics import roc_auc_score


def default_rate(*dfs, date_column='', flag='', legend=[]):
    """
    Plot default rate of your dataframe
    :param dfs: data frames
    :param date_column: name of date column
    :param flag: name of target column
    :param legend: legend for plot
    :return: picture
    """
    plt_ = __import__("matplotlib.pyplot")
    assert len(dfs) == len(legend), 'len(dfs) != len(legend)'
    fig, _ = plt_.pyplot.subplots(figsize=(16, 5))
    for df in dfs:
        DF = df.groupby(date_column).apply(lambda x: x[flag].value_counts())
        DF['DR'] = DF[1]/(DF[1] + DF[0])

        plt_.pyplot.plot(DF.index, DF.DR)

    plt_.pyplot.grid(alpha=0.2)
    plt_.pyplot.legend(legend)
    plt_.pyplot.title('Распределение уровня дефолтов в каждой выборке', fontsize=20)
    plt_.pyplot.show()
    return


def gini_distribution(*dfs, clf=False, date_column='', flag='', score_column='', legend=[], cols_to_drop=[], ylim = []):
    """
    Plot GINI distribution of your model. Have 2 cases:
     1) You have score of your model as a column: use *dfs, date_column, flag, score_column, legend
     2) You have fitted classificator and you need to predict the results: use *dfs, clf, date_column, flag, legend, cols_to_drop
    :param dfs: data frames
    :param clf: classificator
    :param date_column: name of data column
    :param flag: name of target column
    :param score_column: name of score column
    :param legend: list of data frame names for the plot
    :param cols_to_drop: list of column which wasn't in fitting
    :return: picture
    """
    plt_ = __import__("matplotlib.pyplot")
    assert len(dfs) == len(legend), 'len(dfs) != len(legend)'
    fig, _ = plt_.pyplot.subplots(figsize=(16, 5))
    if clf:
        for df in dfs:
            try:
                DF = df.groupby(date_column).apply(lambda x: 2*roc_auc_score(x[flag], clf.predict_proba(x.drop(cols_to_drop, axis=1))[:, 1])-1)
            except Exception:
                DF = df.groupby(date_column).apply(lambda x: 2*roc_auc_score(x[flag], clf.predict(x.drop(cols_to_drop, axis=1)))-1)

            plt_.pyplot.plot(DF)
    else:
        for df in dfs:
            DF = df.groupby(date_column).apply(lambda x: 2 * roc_auc_score(x[flag], x[score_column]) - 1)

            plt_.pyplot.plot(DF)

    plt_.pyplot.grid(alpha=0.2)
    plt_.pyplot.ylabel('GINI', fontsize=20)
    plt_.pyplot.legend(legend, loc='best')
    plt_.pyplot.title('Распределение GINI в каждой выборке', fontsize=20)
    if ylim:
        plt_.pyplot.ylim(ylim)
    plt_.pyplot.show()
    return


def dr_distribution(df_train, df_test, df_tune, research_feature, flag_column, xticks = []):
    """
    Plot default rate distribution of your feature after WOE transformation
    :param research_feature: name of feature
    :param flag_column: name of target column
    :return: picture
    """
    plt_ = __import__("matplotlib.pyplot")

    dff = df_train.copy()
    dff_te = df_test.copy()
    dff_tu = df_tune.copy()

    counts = []
    counts_te = []
    counts_tu = []

    dr = []
    dr_te = []
    dr_tu = []

    vals = sorted(dff[research_feature].unique())
    vals_te = sorted(dff_te[research_feature].unique())
    vals_tu = sorted(dff_tu[research_feature].unique())

    for value in vals:
        sub_df = dff[dff[research_feature] == value]
        counts.append(len(sub_df) / len(dff) * 100)
        dr.append(len(sub_df[sub_df[flag_column] == 1]) / len(sub_df))

    for value in vals_te:
        sub_df = dff_te[dff_te[research_feature] == value]
        counts_te.append(len(sub_df) / len(dff) * 100)
        dr_te.append(len(sub_df[sub_df[flag_column] == 1]) / len(sub_df))

    for value in vals_tu:
        sub_df = dff_tu[dff_tu[research_feature] == value]
        counts_tu.append(len(sub_df) / len(dff) * 100)
        dr_tu.append(len(sub_df[sub_df[flag_column] == 1]) / len(sub_df))

    fig, ax1 = plt_.pyplot.subplots(figsize=(15, 7))
    ax2 = ax1.twinx()

    p1 = ax1.bar(np.arange(len(counts)), counts, width=0.35, color=(24 / 254, 192 / 254, 196 / 254))
    p2 = ax1.bar(np.arange(len(counts)), counts_te, width=0.35, color=(246 / 254, 115 / 254, 109 / 254), align='edge')
    p3 = ax1.bar(np.arange(len(counts)), counts_tu, width=0.35, color=(123 / 254, 197 / 254, 13 / 254), align='edge')

    p1.set_label('Train')
    p2.set_label('Test')
    p3.set_label('Tune')

    ax2.plot(np.arange(len(dr)), dr, marker='o', color='orange')
    ax2.plot(np.arange(len(dr_te)), dr_te, marker='o', color='blue')
    ax2.plot(np.arange(len(dr_tu)), dr_tu, marker='o', color='red')

    for i, v in enumerate(dr):
        ax2.text(i, v + 0.005 * v, str(round(v * 100, 2)) + '%')
    for i, v in enumerate(counts):
        ax1.text(i - 0.1, v - 0.5 * v, str(round(v, 2)) + '%', fontsize=12)
    # ax2.plot(np.arange(len(datas)-2), [gini[2] for gini in ginies_test_sigmoid], marker = 'o', color='red')

    ax1.set_ylabel('Распределение фактора по бакетам (в %)', fontsize=15)
    ax1.set_xlabel('WOE-значение фактора %s' % (research_feature), fontsize=15)
    ax2.set_ylabel('Уровень дефолтов', fontsize=15)

    plt_.pyplot.title('Распределение фактора по бакетам (WOE) и динамика уровня дефолтов', fontsize=15)
    if xticks:
        plt_.pyplot.xticks(np.arange(len(vals)), xticks, rotation=45, fontsize=20)
    else:
        plt_.pyplot.xticks(np.arange(len(vals)), vals, rotation=45, fontsize=20)
    plt_.pyplot.grid(alpha=0.2)
    plt_.pyplot.show()
    return

def add_value_plot(*ginies, feature_names, legend):
    """
    Plot quality of model by adding feature on each iteration
    work in pair with add_value_order function from feature_checker module
    :param ginies: list of ginies
    :param feature_names: list of features whih will add on each iteration
    :param legend: list of data frame names for the plot
    :return:
    """
    plt_ = __import__("matplotlib.pyplot")
    assert len(ginies) == len(legend), 'len(ginies) != len(legend)'

    fig, _ = plt_.pyplot.subplots(figsize=(16, 9))
    for gini in ginies:
        plt_.pyplot.plot(range(1, len(gini) + 1), gini)

    plt_.pyplot.ylabel('GINI', fontsize=20)
    plt_.pyplot.xlabel('features', fontsize=20)
    plt_.pyplot.title('Add value', fontsize=20)
    plt_.pyplot.grid(alpha=0.2)

    plt_.pyplot.legend(legend, loc='lower right')
    plt_.pyplot.xticks(range(1, len(ginies[0]) + 1), feature_names, rotation=90)

    return


def plot_roc_auc_curve(*dfs, clf, cols_to_drop=[], flag_name='def_flag', labels=[]):
    """
    ROC-AUC curve plot
    :param dfs: your data frames for which need to plot
    :param clf: classifier
    :param cols_to_drop: columns which need to drop from your data frame
    :param flag_name: target column
    :param labels: list of labels for legend
    :return:
    """
    plt_ = __import__("matplotlib.pyplot")
    model_name = str(clf)
    model_name = model_name[:model_name.find('(')]

    fig, axes = plt_.pyplot.subplots(1, 2, figsize=(14, 6))
    aucs = []
    for df in dfs:
        X = df.drop(cols_to_drop, axis=1)
        y = df[flag_name]

        try:
            pred_y = clf.predict_proba(X)[:, 1]
        except AttributeError:
            pred_y = clf.predict(X)

        roc_auc = roc_auc_score(y, pred_y)
        fpr, tpr, _ = roc_curve(y, pred_y)
        precision, recall, _ = precision_recall_curve(y, pred_y)
        axes[0].plot(fpr, tpr)
        axes[1].plot(recall, precision)
        aucs.append(roc_auc)

    axes[0].plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))

    axes[0].grid(alpha=0.2)
    axes[0].legend(
        ['auc on %s = %.3f' % (l, auc) for i, l in enumerate(labels) for j, auc in enumerate(aucs) if i == j],
        loc="lower right")
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")
    axes[0].set_title("%s ROC curve" % (model_name))

    axes[1].grid(alpha=0.2)
    axes[1].legend(labels, loc='best')
    axes[1].set_xlabel("recall")
    axes[1].set_ylabel("precision")
    axes[1].set_title("Precision-Recall curve")
    plt_.pyplot.tight_layout()

    return
