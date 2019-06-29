from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np
import statsmodels.formula.api as smf
import pandas as pd


def get_vif(exogs, data):
    """
    Return VIF (variance inflation factor) DataFrame

    Args:
    exogs (list): list of exogenous/independent variables
    data (DataFrame): the df storing all variables

    Returns:
    VIF and Tolerance DataFrame for each exogenous variable

    Notes:
    Assume we have a list of exogenous variable [X1, X2, X3, X4].
    To calculate the VIF and Tolerance for each variable, we regress
    each of them against other exogenous variables. For instance, the
    regression model for X3 is defined as:
                        X3 ~ X1 + X2 + X4
    And then we extract the R-squared from the model to calculate:
                    VIF = 1 / (1 - R-squared)
                    Tolerance = 1 - R-squared
    The cutoff to detect multicollinearity:
                    VIF > 10 or Tolerance < 0.2
    """

    # initialize arrays
    vif_array = np.array([])
    tolerance_array = np.array([])
    # create formula for each exogenous variable
    for exog in exogs:
        not_exog = [i for i in exogs if i != exog]
        formula = f"{exog} ~ {' + '.join(not_exog)}"
        # extract r-squared from the fit
        r_squared = smf.ols(formula, data=data).fit().rsquared
        # calculate VIF
        vif = 1/(1-r_squared)
        vif_array = np.append(vif_array, vif).round(2)
        # calculate tolerance
        tolerance = 1-r_squared
        tolerance_array = np.append(tolerance_array, tolerance)
    # return VIF DataFrame
    df_vif = pd.DataFrame({'VIF': vif_array, 'Tolerance': tolerance_array}, index=exogs)
    return df_vif


def get_iv(woe_df):
    """
    Function to calculate IV by WOE
    You should have a df with transformed features by WOE
    All features names starts with 'WOE_'
    :param woe_df: your dataFrame with WOE features
    :return: df[feature, iv]
    """

    def event(x):
        return 1 / (1 + np.exp(-x))

    def nonevent(x):
        return 1 / (1 + np.exp(x))
    data = []
    for feature in woe_df:
        for woe in woe_df[feature].unique():
            ivs = []
            ivs.append((event(woe) - nonevent(woe)) * woe)
        data.append((feature.split('WOE_')[1], np.sum(ivs)))
        ivs = []

    iv_df = pd.DataFrame(data, columns=['feature', 'iv'])
    return iv_df


def calculate_psi(expected, actual, buckettype='bins', buckets=10, axis=0):
    """
    Calculate the PSI (population stability index) across all variables

    Args:
       expected: numpy matrix of original values
       actual: numpy matrix of new values, same size as expected
       buckettype: type of strategy for creating buckets, bins splits into even splits, quantiles splits into quantile buckets
       buckets: number of quantiles to use in bucketing variables
       axis: axis by which variables are defined, 0 for vertical, 1 for horizontal

    Returns:
       psi_values: ndarray of psi values for each variable

    Author:
       Matthew Burke
       github.com/mwburke
       worksofchart.com
    """

    def psi(expected_array, actual_array, buckets):
        """Calculate the PSI for a single variable

        Args:
           expected_array: numpy array of original values
           actual_array: numpy array of new values, same size as expected
           buckets: number of percentile ranges to bucket the values into

        Returns:
           psi_value: calculated PSI value
        """

        def scale_range (input, min, max):
            input += -(np.min(input))
            input /= np.max(input) / (max - min)
            input += min
            return input


        breakpoints = np.arange(0, buckets + 1) / (buckets) * 100

        if buckettype == 'bins':
            breakpoints = scale_range(breakpoints, np.min(expected_array), np.max(expected_array))
        elif buckettype == 'quantiles':
            breakpoints = np.stack([np.percentile(expected_array, b) for b in breakpoints])



        expected_percents = np.histogram(expected_array, breakpoints)[0] / len(expected_array)
        actual_percents = np.histogram(actual_array, breakpoints)[0] / len(actual_array)

        def sub_psi(e_perc, a_perc):
            """Calculate the actual PSI value from comparing the values.
               Update the actual value to a very small number if equal to zero
            """
            if a_perc == 0:
                a_perc = 0.0001
            if e_perc == 0:
                e_perc = 0.0001

            value = (e_perc - a_perc) * np.log(e_perc / a_perc)
            return(value)

        psi_value = np.sum(sub_psi(expected_percents[i], actual_percents[i]) for i in range(0, len(expected_percents)))

        return(psi_value)

    if len(expected.shape) == 1:
        psi_values = np.empty(len(expected.shape))
    else:
        psi_values = np.empty(expected.shape[axis])

    for i in range(0, len(psi_values)):
        if len(psi_values) == 1:
            psi_values = psi(expected, actual, buckets)
        elif axis == 0:
            psi_values[i] = psi(expected[:,i], actual[:,i], buckets)
        elif axis == 1:
            psi_values[i] = psi(expected[i,:], actual[i,:], buckets)

    return(psi_values)


def add_value_order(df_train, df_test, df_tune, cols_to_drop, target_column, clf):
    """
    return list of features sorted by add value
    :param target_column: name of target column
    :param clf: Classificator
    :return: sorted list of features
    """

    columns = df_train.drop(cols_to_drop, axis=1).columns
    # new_order = []
    final_list = [('feature', 0, 0, 0)]
    gginies = []

    model = clf
    while len(columns) != 0:
        for i, feature in enumerate(tqdm(columns)):
            model.fit(df_train[new_order + [feature]].values, df_train[target_column])

            y_pred_train = model.predict_proba(df_train[new_order + [feature]])[:, 1]
            y_pred_test = model.predict_proba(df_test[new_order + [feature]])[:, 1]
            y_pred_tune = model.predict_proba(df_tune[new_order + [feature]])[:, 1]

            gini_train = (2*roc_auc_score(df_train.def_flag, y_pred_train) - 1)*100
            gini_test = (2*roc_auc_score(df_test.def_flag, y_pred_test) - 1)*100
            gini_tune = (2*roc_auc_score(df_tune.def_flag, y_pred_tune) - 1)*100

            gginies.append((feature, gini_train, gini_test, gini_tune))

        g_features = [t for t in gginies if (t[1] > final_list[-1][1]) and (t[3] > final_list[-1][3])]
        if g_features == []:
            columns = []
        else:
            final_list.append(sorted(g_features, key=lambda x: x[1], reverse=True)[0])
            # new_order.append(final_list[-1][0])
            columns = [feature for feature in columns if feature not in new_order]
            gginies = []
    return final_list[1:]


def two_bucket_dr(df_train, df_test, df_tune=False, target_column='def_flag', cdp=[], tol=0.05):
    """
    Calculate default rate for each feature if it binned on two buckets

    Example:

    dr, s_f = get_dr_df(df_train=df_train,
               df_test=df_test,
               df_tune=df_tune,
               cdp=cols_to_drop)

    ll = []
    bad = []
    for f in tqdm(dr.keys()):
        try:
            if all(dr[f]) > 0:
                ll.append((f, dr[f]))
        except:
            bad.append(f)
            continue

    additionally_features = [f for f in ll if (f[1][0][0]/f[1][0][1] >= 1.5 or f[1][0][0]/f[1][0][1] <= 0.66) and \
                                              (f[1][2][0]/f[1][2][1] >= 1.5 or f[1][2][0]/f[1][2][1] <= 0.66)]

    :param target_column: name of target column
    :param cdp: columns to drop
    :param tol: tolerance
    :return:
    """
    dff_tr = df_train.copy()
    dff_te = df_test.copy()
    if df_tune:
        dff_tu = df_tune.copy()

    dr = {}
    suspicious_features = []

    for feature in tqdm(dff_tr.drop(cdp, axis=1).columns):

        wing = WingsOfEvidence(columns_to_apply=[feature], is_monotone=True, verbose=False, \
                               n_initial=2, n_target=2, optimizer='full-search', only_values=True)
        try:
            wing.fit(dff_tr, y=dff_tr[target_column])

            dff_tr_woe = wing.transform(dff_tr)
            dff_te_woe = wing.transform(dff_te)
            if df_tune:
                dff_tu_woe = wing.transform(dff_tu)

            dff_tr_woe = pd.concat([dff_tr[target_column], dff_tr_woe], axis=1)
            dff_te_woe = pd.concat([dff_te[target_column], dff_te_woe], axis=1)
            if df_tune:
                dff_tu_woe = pd.concat([dff_tu[target_column], dff_tu_woe], axis=1)

            dr_tr = []
            dr_te = []
            if df_tune:
                dr_tu = []

            vals_tr = sorted(dff_tr_woe['WOE_' + feature].unique())
            vals_te = sorted(dff_te_woe['WOE_' + feature].unique())
            if df_tune:
                vals_tu = sorted(dff_tu_woe['WOE_' + feature].unique())

            for value in vals_tr:
                sub_df = dff_tr_woe[dff_tr_woe['WOE_' + feature] == value]
                dr_tr.append(len(sub_df[sub_df[target_column] == 1]) / len(sub_df))

            for value in vals_te:
                sub_df = dff_te_woe[dff_te_woe['WOE_' + feature] == value]
                dr_te.append(len(sub_df[sub_df[target_column] == 1]) / len(sub_df))

            if df_tune:
                for value in vals_tu:
                    sub_df = dff_tu_woe[dff_tu_woe['WOE_' + feature] == value]
                    dr_tu.append(len(sub_df[sub_df[target_column] == 1]) / len(sub_df))

            incr = 0
            decr = 0
            if df_tune:
                for dr_checked in [dr_tr, dr_te, dr_tu]:
                    check_increase = [x < y * (1 - tol) for x, y in zip(dr_checked, dr_checked[1:])]
                    if all(check_increase) == True:
                        incr += 1
                    check_decrease = [x > y * (1 + tol) for x, y in zip(dr_checked, dr_checked[1:])]
                    if all(check_decrease) == True:
                        decr += 1
            else:
                for dr_checked in [dr_tr, dr_te]:
                    check_increase = [x < y * (1 - tol) for x, y in zip(dr_checked, dr_checked[1:])]
                    if all(check_increase) == True:
                        incr += 1
                    check_decrease = [x > y * (1 + tol) for x, y in zip(dr_checked, dr_checked[1:])]
                    if all(check_decrease) == True:
                        decr += 1

            if incr == 3 or decr == 3:
                if df_tune:
                    dr[feature] = [dr_tr, dr_te, dr_tu]
                else:
                    dr[feature] = [dr_tr, dr_te]

        except Exception:
            suspicious_features.append(feature)
            continue

    return dr, suspicious_features


def get_gini(df, clf, columns, target):
    """
    function to calculate gini for each column in your dataframe
    :param df: dataFrame
    :param clf: classificator
    :param columns: list of columns
    :param target: target column name
    :return: gini dict
    """
    gini_dict = {}
    t = tqdm_notebook(columns, leave=False)
    for column in t:
        t.set_description(f'{column} ')
        # sub_df = df[[target, column]]
        clf.fit(df[[column]], df[target])
        try:
            y_pred_train = clf.predict_proba(df[[column]])[:, 1]
        except Exception:
            y_pred_train = clf.predict(df[[column]])

        roc_auc_train = roc_auc_score(df[target], y_pred_train)
        gini_dict[column] = (2 * roc_auc_train - 1) * 100

    return gini_dict