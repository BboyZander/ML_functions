import operator
import tqdm
from wing import WingsOfEvidence
import pandas as pd

def get_bad_features(df_train_woe, df_test_woe, df_tune_woe, tol=0.05, use_test=True):
    '''
    Function which helps you find features with non monotonic distribution by bucket
    :param tol: tolerance
    :param use_test: do we look on test df or not
    :return: list with bad features
    '''
    bad_features = []

    for feature in df_train_woe.drop('def_flag', axis=1).columns:
        dr = []
        dr_te = []
        dr_tu = []

        vals = sorted(df_train_woe[feature].unique())
        vals_te = sorted(df_test_woe[feature].unique())
        vals_tu = sorted(df_tune_woe[feature].unique())

        for value in vals:
            sub_df = df_train_woe[df_train_woe[feature] == value]
            dr.append(len(sub_df[sub_df.def_flag == 1]) / len(sub_df))

        for value in vals_te:
            sub_df = df_test_woe[df_test_woe[feature] == value]
            dr_te.append(len(sub_df[sub_df.def_flag == 1]) / len(sub_df))

        for value in vals_tu:
            sub_df = df_tune_woe[df_tune_woe[feature] == value]
            dr_tu.append(len(sub_df[sub_df.def_flag == 1]) / len(sub_df))

        incr = 0
        decr = 0

        if use_test:
            for dr_checked in [dr, dr_te, dr_tu]:
                check_increase = [x < y * (1 - tol) for x, y in zip(dr_checked, dr_checked[1:])]
                if all(check_increase) == True:
                    incr += 1
                check_decrease = [x > y * (1 + tol) for x, y in zip(dr_checked, dr_checked[1:])]
                if all(check_decrease) == True:
                    decr += 1

            if incr != 3 and decr != 3:
                bad_features.append(feature)
        else:
            for dr_checked in [dr, dr_tu]:
                check_increase = [x < y * (1 - tol) for x, y in zip(dr_checked, dr_checked[1:])]
                if all(check_increase) == True:
                    incr += 1
                check_decrease = [x > y * (1 + tol) for x, y in zip(dr_checked, dr_checked[1:])]
                if all(check_decrease) == True:
                    decr += 1

            if incr != 2 and decr != 2:
                bad_features.append(feature)

    return bad_features

def woe_rebinning(df_train, df_test, df_oot, df_tune, cols_to_drop, bad_features, tol=0.05, use_test=True):
    '''
    Find best splitting for feature by buckets with monotonic distribution using wings module

    :param cols_to_drop:
    :param bad_features: list of bad features
    :param tol: tolerance
    :param use_test: do we look on test df or not
    :return: list with dataframes with new woe transformation
    '''
    drop_features = []
    df_rebinned = []
    for woe_feature in tqdm(bad_features):
        ginies = {}
        dfs = {}

        for n_initial in range(2, 11):
            feature = woe_feature.split('WOE_')[1]
            wings_0 = WingsOfEvidence(columns_to_apply=[feature], verbose=False, is_monotone=True, \
                                      n_initial=n_initial, n_target=2, optimizer='full-search', only_values=True)

            wings_0.fit(df_train, y=df_train.def_flag)

            df_train_woe_bad = wings_0.transform(df_train)
            df_test_woe_bad = wings_0.transform(df_test)
            df_oot_woe_bad = wings_0.transform(df_oot)
            df_tune_woe_bad = wings_0.transform(df_tune)

            df_train_woe_bad = pd.concat([df_train[cols_to_drop], df_train_woe_bad], axis=1)
            df_test_woe_bad = pd.concat([df_test[cols_to_drop], df_test_woe_bad], axis=1)
            df_oot_woe_bad = pd.concat([df_oot[cols_to_drop], df_oot_woe_bad], axis=1)
            df_tune_woe_bad = pd.concat([df_tune[cols_to_drop], df_tune_woe_bad], axis=1)

            dr = []
            dr_te = []
            dr_tu = []

            vals = sorted(df_train_woe_bad[woe_feature].unique())
            vals_te = sorted(df_test_woe_bad[woe_feature].unique())
            vals_tu = sorted(df_tune_woe_bad[woe_feature].unique())

            for value in vals:
                sub_df = df_train_woe_bad[df_train_woe_bad[woe_feature] == value]
                dr.append(len(sub_df[sub_df.def_flag == 1]) / len(sub_df))

            for value in vals_te:
                sub_df = df_test_woe_bad[df_test_woe_bad[woe_feature] == value]
                dr_te.append(len(sub_df[sub_df.def_flag == 1]) / len(sub_df))

            for value in vals_tu:
                sub_df = df_tune_woe_bad[df_tune_woe_bad[woe_feature] == value]
                dr_tu.append(len(sub_df[sub_df.def_flag == 1]) / len(sub_df))

            incr = 0
            decr = 0

            if use_test:
                for dr_checked in [dr, dr_te, dr_tu]:
                    check_increase = [x < y * (1 - tol) for x, y in zip(dr_checked, dr_checked[1:])]
                    if all(check_increase) == True:
                        incr += 1
                    check_decrease = [x > y * (1 + tol) for x, y in zip(dr_checked, dr_checked[1:])]
                    if all(check_decrease) == True:
                        decr += 1

                if incr == 3 or decr == 3:
                    ginies[n_initial] = wings_0.gini_dict[feature]
                    dfs[n_initial] = (df_train_woe_bad, df_test_woe_bad, df_oot_woe_bad, df_tune_woe_bad)
            else:
                for dr_checked in [dr, dr_tu]:
                    check_increase = [x < y * (1 - tol) for x, y in zip(dr_checked, dr_checked[1:])]
                    if all(check_increase) == True:
                        incr += 1
                    check_decrease = [x > y * (1 + tol) for x, y in zip(dr_checked, dr_checked[1:])]
                    if all(check_decrease) == True:
                        decr += 1

                if incr == 2 or decr == 2:
                    ginies[n_initial] = wings_0.gini_dict[feature]
                    dfs[n_initial] = (df_train_woe_bad, df_test_woe_bad, df_oot_woe_bad, df_tune_woe_bad)

        if len(ginies) == 0:
            drop_features.append(woe_feature)
        else:
            max_n_init = max(ginies.items(), key=operator.itemgetter(1))[0]
            #             max_n_init = max(ginies)
            #             print(max_n_init)
            df_rebinned.append(dfs[max_n_init])

    return df_rebinned, drop_features

def split_from_rebinned(*df_woe, df_rebinned, good_features=[]):
    '''
    Join old df with good features with new features after rebinning
    :param df_rebinned: result of
    :param good_features:
    :return: concated df
    '''
    results = []
    for df in df_woe:
        df = df[good_features]
        for ind in range(len(df_rebinned)):
            df = pd.concat([df, df_rebinned[ind][0].iloc[:, -1]], axis=1)
        results.append(df)
    if len(results) == 1:
        return results[0]
    else:
        return results