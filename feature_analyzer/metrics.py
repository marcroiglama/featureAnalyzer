from tqdm import tqdm
from collections import Counter
import numpy as np
import pandas as pd
import pandas.api.types as pandas_types
import scipy.stats as ss
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import OrdinalEncoder


GRAPHICS = False


def heatmap(df, title):
    if GRAPHICS:
        idx, col = df.shape
        heigh = 8
        weigth = 8
        if idx > 20:
            heigh = 12
        plt.figure(figsize=(weigth, heigh))
        ax = sns.heatmap(df, annot=False, cmap=plt.cm.Blues, xticklabels=True, yticklabels=True)
        plt.title(title)
        plt.tight_layout()
        plt.show()


def _correlation_sort(df_corr):
    if len(df_corr.columns) > 1:
        ordered_correlations = df_corr.abs().where(
            np.triu(np.ones(df_corr.abs().shape), k=1).astype(np.bool)).stack().sort_values(ascending=False)
        ordered_correlations.index = ordered_correlations.index.map('{0[0]}//{0[1]}'.format)
    else:
        ordered_correlations = df_corr[df_corr.columns[0]].sort_values(ascending=False)
        ordered_correlations.index = ordered_correlations.index.map(lambda x: "{}//{}".format(df_corr.columns[0], x))
    return ordered_correlations


# NUMERIC vs NUMERIC
def correlation(features, labels):
    correlations = pd.DataFrame(index=features.columns)
    for y in tqdm(labels.columns):
        correlations[y] = features.corrwith(labels[y])
    heatmap(
        correlations[(correlations.abs() >= 0.2) & (correlations.abs() < 0.99)].dropna(axis=1, how='all').dropna(axis=0,
                                                                                                                 how='all'),
        "NUMERIC CORRELATIONS")
    correlations = _correlation_sort(correlations)
    return correlations


# NUMERIC vs CATEGORY
def _correlation_ratio(x_num, y_cat):
    fcat, _ = pd.factorize(y_cat)
    cat_num = np.max(fcat) + 1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0, cat_num):
        cat_measures = x_num[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
    numerator = np.sum(np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2)))
    denominator = np.sum(np.power(np.subtract(x_num, y_total_avg), 2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator / denominator)
    return eta


def compute_correlation_ratio(x_nums, y_cats):
    corr_ratio = pd.DataFrame(index=x_nums.columns)
    for y_cat in tqdm(y_cats.columns):
        corr_ratio[y_cat] = [_correlation_ratio(x_nums[x_num], y_cats[y_cat]) for x_num in x_nums]
    heatmap(corr_ratio, "CATEGORIC CORRELATION RATIO")
    corr_ratio = _correlation_sort(corr_ratio)
    return corr_ratio


def _get_groups_by_label(x_cats, x_cat, y_nums, y_num):
    groups = x_cats[x_cat].unique()
    try:
        groups = groups[~np.isnan(groups)]
    except:
        pass
    return [y_nums.loc[x_cats[x_cat] == group, y_num] for group in groups]


def compute_anova(x_cats, y_nums):
    significance_threshold = 0.95
    anova = pd.DataFrame(index=x_cats.columns)
    for y_num in tqdm(y_nums.columns):
        temp = []
        for x_cat in x_cats:
            x_groups = _get_groups_by_label(x_cats, x_cat, y_nums, y_num)
            temp.append(ss.f_oneway(*x_groups)[1])
        anova[y_num] = temp
    # anova[anova >= significance_threshold] = 1
    # anova[anova < significance_threshold] = 0
    heatmap(anova, "ANOVA")
    return anova


def compute_kruskal(x_cats, y_nums):
    significance_threshold = 0.95
    kruskal = pd.DataFrame(index=x_cats.columns)
    for y_num in tqdm(y_nums.columns):
        temp = []
        for x_cat in x_cats:
            x_groups = _get_groups_by_label(x_cats, x_cat, y_nums, y_num)
            temp.append(1 - ss.kruskal(*x_groups)[1])
        kruskal[y_num] = temp
    # kruskal[kruskal >= significance_threshold] = 1
    # kruskal[kruskal < significance_threshold] = 0
    heatmap(kruskal, "KRUSKAL")
    return kruskal


def _dimension_reduction(features, labels, method):
    x = features.columns
    if len(x) > 2:
        reduced_features = method.fit_transform(features)
        reduced_features = pd.DataFrame(reduced_features, columns=['dimension_1', 'dimension_2'], index=features.index)
        for y in tqdm(labels.columns):
            reduced_features[y] = labels[y]
            plt.figure(figsize=(20, 15))
            sns.lmplot(x='dimension_1', y='dimension_2', data=reduced_features, hue=y, fit_reg=False)

    elif len(x) == 2:
        features = features.copy()
        for y in labels:
            features[y] = labels[y]
            sns.lmplot(x=x[0], y=x[1], data=features, hue=y, fit_reg=False)
    plt.show()


def tsne(features, labels):
    if GRAPHICS:
        tsne = TSNE(n_components=2)
        _dimension_reduction(features, labels, tsne)


def pca(features, labels):
    if GRAPHICS:
        pca = PCA(n_components=2)
        _dimension_reduction(features, labels, pca)


# CATEGORY vs CATEGORY
def _cramers(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def compute_cramers(features, labels):
    cramers_corr = pd.DataFrame(index=features.columns)
    for y in tqdm(labels.columns):
        cramers_corr[y] = [_cramers(features[x], labels[y]) for x in features.columns]
    heatmap(cramers_corr, "CRAMER'S")
    return cramers_corr


def _conditional_entropy(x, y):
    # entropy of x given y
    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x, y)))
    total_occurrences = sum(y_counter.values())
    entropy = 0
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        entropy += p_xy * np.log(p_y / p_xy)
    return entropy


def _theil_u(x, y):
    s_xy = _conditional_entropy(x, y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n / total_occurrences, x_counter.values()))
    s_x = ss.entropy(p_x)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x


def compute_theil_u(x_cats, y_cats):
    theil_u = pd.DataFrame(index=x_cats.columns)
    for y_cat in tqdm(y_cats.columns):
        theil_u[y_cat] = [_theil_u(x_cats[x_cat].tolist(), y_cats[y_cat].tolist()) for x_cat in x_cats]
    heatmap(theil_u, "THEIL'S U")
    return theil_u


def dummy_corr(x_cats, y_cats):
    for y_cat in tqdm(y_cats.columns):
        if (len(y_cats[y_cat].unique()) == 2) & (1 in y_cats[y_cat].unique()) & (0 in y_cats[y_cat].unique()):
            already_dummy = True
        else:
            already_dummy = False
        dummy_matrix = pd.get_dummies(y_cats[y_cat], prefix=y_cat) if not already_dummy else y_cats[y_cat]
        for x_cat in x_cats:
            x_dummy = pd.get_dummies(x_cats[x_cat], prefix=x_cat)
            dummy_matrix = dummy_matrix.join(x_dummy)
        dummy_corr = dummy_matrix.corr()
        heatmap(
            dummy_corr[(dummy_corr.abs() >= 0.2) & (dummy_corr.abs() < 0.99)].dropna(axis=1, how='all').dropna(axis=0,
                                                                                                               how='all'),
            "DUMMY CORRELATION")
    return dummy_corr


# ALL vs ONE
def randomforest_importances(features, label, topn=10):
    if (pandas_types.is_categorical_dtype(label)) | (pandas_types.is_object_dtype(label)):
        rdf = RandomForestClassifier(class_weight='balanced')
    else:
        rdf = RandomForestRegressor()

    rdf.fit(features, label)
    importances = rdf.feature_importances_
    forest_importances = dict(zip(features.columns, importances))

    indices = np.argsort(importances)[0:topn]
    if GRAPHICS:
        plt.figure(figsize=(10, 5))
        plt.title('Feature Importances')
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [features.columns[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.show()

    return forest_importances


