import pandas as pd

from feature_analyzer import metrics


class FeatureAnalyzer():

    def __init__(self, df: pd.DataFrame, numerical_features: list, numerical_labels: list, categorical_features: list,
                 categorical_labels: list, graphics: bool = False):
        self.df = df.copy().reset_index()
        self.categorical_features = categorical_features
        self.categorical_labels = categorical_labels
        self.numerical_features = numerical_features
        self.numerical_labels = numerical_labels
        self.all_columns = categorical_features + categorical_labels + numerical_features + numerical_labels

        self.df_cat_features = self.df[categorical_features]
        self.df_cat_labels = self.df[categorical_labels]
        self.df_num_features = self.df[numerical_features]
        self.df_num_labels = self.df[numerical_labels]

        self._set_df_dtypes()

        metrics.GRAPHICS = graphics

    def _set_df_dtypes(self):
        self.df = self.df[self.all_columns]
        self.df[self.categorical_features + self.categorical_labels].astype("category", copy=False)
        self.df[self.numerical_features + self.numerical_labels].astype("float", copy=False)

    @staticmethod
    def _check_column_exists(list_one, list_two):
        return True if (len(list_one) != 0) & (len(list_two) != 0) else False

    def num_vs_num(self):
        features = self.df_num_features
        labels = self.df_num_labels
        if self._check_column_exists(features.columns, labels.columns):
            df_corr = metrics.correlation(features, labels)
        else:
            df_corr = None
        return df_corr

    def num_vs_cat(self):
        features = self.df_num_features
        labels = self.df_cat_labels
        if self._check_column_exists(features.columns, labels.columns):
            df_corr_ratio = metrics.compute_correlation_ratio(features, labels)
            metrics.pca(features, labels)
        else:
            df_corr_ratio = None
        return df_corr_ratio

    def cat_vs_num(self):
        features = self.df_cat_features
        labels = self.df_num_labels
        if self._check_column_exists(features.columns, labels.columns):
            df_anova = metrics.compute_anova(features, labels)
            df_kruskal = metrics.compute_kruskal(features, labels)
        else:
            df_anova = None
            df_kruskal = None
        return df_anova, df_kruskal

    def cat_vs_cat(self):
        features = self.df_cat_features
        labels = self.df_cat_labels
        if self._check_column_exists(features.columns, labels.columns):
            df_dummy_corr = metrics.dummy_corr(features, labels)
            df_cramers = metrics.compute_cramers(features, labels)
            df_theil = metrics._theil_u(features, labels)
        else:
            df_dummy_corr = None
            df_cramers = None
            df_theil = None
        return df_dummy_corr, df_cramers, df_theil

    def random_forest_relevances(self, features: list, label: str):
        return metrics.randomforest_importances(features, label)

    def abstract(self):
        report = {}
        report["correlation"] = self.num_vs_num().to_dict()
        report["correlation_ration"] = self.num_vs_cat().to_dict()
        report["anova"], report["kruscal"] = (x.to_dict() for x in self.cat_vs_num())
        dummy_corr, cramers, theil = self.cat_vs_cat()

        report["dummy_corr"], report["cramers"], report["theil"] = dummy_corr.to_dict(), cramers.to_dict(), theil
        return report
