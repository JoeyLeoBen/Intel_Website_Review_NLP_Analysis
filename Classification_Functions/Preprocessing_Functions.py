from typing import Any, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from patsy import dmatrices
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Set pandas display options
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)


class Preprocessing:
    @staticmethod
    def num_univariate_histogram(
        df: pd.DataFrame,
        length: float,
        width: float,
        rows: int,
        col: int,
        font: float,
        kind: int,
    ) -> None:
        """
        Plot histograms for all numeric columns in the DataFrame and display descriptive statistics.

        Depending on the 'kind' parameter, the function selects all columns except the last one (kind == 1)
        or uses all columns (kind == 2).

        Args:
            df (pd.DataFrame): DataFrame containing numeric data.
            length (float): Figure height for the plot.
            width (float): Figure width for the plot.
            rows (int): Number of rows in the histogram grid layout.
            col (int): Number of columns in the histogram grid layout.
            font (float): Font scaling factor for the plots.
            kind (int): Determines which numeric columns to plot (1: all except last, 2: all columns).

        Returns:
            None
        """
        if kind == 1:
            X_num = df[df.columns[0:-1]]
            sns.set(font_scale=font, style="white")
            X_num.hist(bins=50, figsize=(width, length), layout=(rows, col), grid=False)
            plt.show()
            print("\nX continuous descriptive stats:")
            describe = X_num.describe().T
            display(describe)

        if kind == 2:
            X_num = df.copy()
            sns.set(font_scale=font, style="white")
            X_num.hist(bins=50, figsize=(width, length), layout=(rows, col), grid=False)
            plt.show()
            print("\nX continuous descriptive stats:")
            describe = X_num.describe().T
            display(describe)

        return None

    @staticmethod
    def cat_univariate_freq(
        df: pd.DataFrame,
        length: float,
        width: float,
        col_start: int,
        col_end: int,
        font: float,
    ) -> None:
        """
        Plot frequency bar charts for all categorical variables in the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing categorical data.
            length (float): Figure height for the plot.
            width (float): Figure width for the plot.
            col_start (int): Starting index for selecting categorical columns.
            col_end (int): Ending index for selecting categorical columns.
            font (float): Font scaling factor for the plots.

        Returns:
            None
        """
        X_cat = df.select_dtypes(include=["object"]).columns[col_start:col_end]
        for X in X_cat:
            series = round((df[X].value_counts(normalize=True)) * 100, 0)
            series = series.sort_values(ascending=True)
            sns.set(font_scale=font, style="white")
            series.plot.barh(figsize=(width, length))
            plt.title(f"{X} frequencies")
            plt.xlabel("percent")
            plt.ylabel(X)
            plt.show()

        return None

    @staticmethod
    def target_univariate_scatter(
        df: pd.DataFrame,
        x: str,
        y: str,
        length: float,
        width: float,
        font: float,
    ) -> None:
        """
        Generate a scatter plot for two specified columns along with basic supporting statistics.

        Args:
            df (pd.DataFrame): DataFrame containing the data.
            x (str): Column name for the x-axis.
            y (str): Column name for the y-axis.
            length (float): Figure height for the plot.
            width (float): Figure width for the plot.
            font (float): Font scaling factor for the plot.

        Returns:
            None
        """
        df = df.reset_index()
        sns.set(font_scale=font, style="white")
        plt.figure(figsize=(width, length))
        sns.scatterplot(data=df, x=x, y=y)
        plt.title("season " + y + " by " + x)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.show()

        return None

    @staticmethod
    def num_bivariate_scatter(
        df: pd.DataFrame,
        y: Union[str, List[str]],
        x: Union[str, List[str]],
        font: float,
        length: float,
        width: float,
        dot_size: float,
    ) -> None:
        """
        Generate scatter plot(s) for numeric variables using pairplot.

        Args:
            df (pd.DataFrame): DataFrame containing numeric data.
            y (Union[str, List[str]]): Column(s) to be used for y-axis.
            x (Union[str, List[str]]): Column(s) to be used for x-axis.
            font (float): Font scaling factor for the plots.
            length (float): Figure height for the plot.
            width (float): Figure width for the plot.
            dot_size (float): Size of the dots in the scatter plots.

        Returns:
            None
        """
        sns.set(font_scale=font, style="white")
        plot = sns.pairplot(
            data=df,
            y_vars=y,
            x_vars=x,
            diag_kind=None,
            plot_kws={"s": dot_size},
        )
        plot.fig.set_size_inches(width, length)
        plt.show()

        return None

    @staticmethod
    def num_bivariate_corr_target(
        df: pd.DataFrame,
        target: str,
        threshold: float,
        font: float,
        length: float,
        width: float,
    ) -> None:
        """
        Generate a correlation heatmap between all numeric features and the target variable.
        Also displays features with correlation less than a given threshold.

        Args:
            df (pd.DataFrame): DataFrame containing numeric features and a target variable.
            target (str): Target column name.
            threshold (float): Threshold value for filtering features.
            font (float): Font scaling factor for the plot.
            length (float): Figure height for the plot.
            width (float): Figure width for the plot.

        Returns:
            None
        """
        X_corr = df.corr(method="pearson")
        X_corr = X_corr[[target]].sort_values(by=[target], ascending=False)
        sns.set(font_scale=font, style="white")
        fig, ax = plt.subplots()
        fig.set_size_inches(width, length)
        sns.heatmap(X_corr, ax=ax)
        plt.title("correlation matrix")
        plt.show()
        display(X_corr)
        X_corr = X_corr.reset_index()
        X_corr[target] = abs(X_corr[target])
        X_corr = X_corr.loc[X_corr[target] < threshold]
        X_corr = list(X_corr["index"])
        print("\nfeatures to remove: ")
        print(X_corr)

        return None

    @staticmethod
    def cat_bivariate_avg_target(
        df: pd.DataFrame,
        col_start: int,
        col_end: int,
        target: str,
        length: float,
        width: float,
        font: float,
    ) -> None:
        """
        Generate bar plots to visualize the average of a numeric target variable for different categorical groups.

        Args:
            df (pd.DataFrame): DataFrame containing the data.
            col_start (int): Starting index for selecting categorical columns.
            col_end (int): Ending index for selecting categorical columns.
            target (str): Numeric target column for which to compute averages.
            length (float): Figure height for the plot.
            width (float): Figure width for the plot.
            font (float): Font scaling factor for the plots.

        Returns:
            None
        """
        X_cat = df.select_dtypes(include=["object"]).columns[col_start:col_end]
        for X in X_cat:
            label = df[[X, target]].sort_values(by=[target], ascending=False)
            label = label.groupby([X]).mean().sort_values(by=[target], ascending=True)
            label["positive"] = label[target] > 0
            sns.set(font_scale=font, style="white")
            label[target].plot(
                kind="barh",
                figsize=(width, length),
                color=label.positive.map({True: "b", False: "r"}),
            )
            plt.title("average " + target + " per " + X)
            plt.xlabel("average " + target)
            plt.ylabel(X)
            plt.show()
            label = label.sort_values(by=[target], ascending=False)
            display(label)

        return None

    @staticmethod
    def cat_bivariate_sum_target(
        df: pd.DataFrame,
        col_start: int,
        col_end: int,
        target: str,
        length: float,
        width: float,
        font: float,
    ) -> None:
        """
        Generate bar plots to visualize the sum of a numeric target variable for different categorical groups.

        Args:
            df (pd.DataFrame): DataFrame containing the data.
            col_start (int): Starting index for selecting categorical columns.
            col_end (int): Ending index for selecting categorical columns.
            target (str): Numeric target column for which to compute sums.
            length (float): Figure height for the plot.
            width (float): Figure width for the plot.
            font (float): Font scaling factor for the plots.

        Returns:
            None
        """
        X_cat = df.select_dtypes(include=["object"]).columns[col_start:col_end]
        for X in X_cat:
            label = df[[X, target]].sort_values(by=[target], ascending=False)
            label = label.groupby([X]).sum().sort_values(by=[target], ascending=True)
            label["positive"] = label[target] > 0
            sns.set(font_scale=font, style="white")
            label[target].plot(
                kind="barh",
                figsize=(width, length),
                color=label.positive.map({True: "b", False: "r"}),
            )
            plt.title("average " + target + " per " + X)
            plt.xlabel("average " + target)
            plt.ylabel(X)
            plt.show()
            label = label.sort_values(by=[target], ascending=False)
            display(label)

        return None

    @staticmethod
    def remove_outliers(df: pd.DataFrame, col: str) -> Tuple[str, float, str, float]:
        """
        Remove outliers from a given column using the IQR method.

        Args:
            df (pd.DataFrame): DataFrame containing the data.
            col (str): Column name from which to remove outliers.

        Returns:
            Tuple[str, float, str, float]: A tuple containing a label and the lower outlier bound,
                                             and a label and the upper outlier bound.
        """
        p_25 = df[col].quantile(0.25)
        p_75 = df[col].quantile(0.75)
        iqr = (p_75 - p_25) * 1.5
        low_outliers = p_25 - iqr
        high_outliers = p_75 + iqr
        df = df.loc[(df[col] > low_outliers) & (df[col] < high_outliers)]

        return ("low end outliers:", low_outliers, "high end outliers", high_outliers)

    @staticmethod
    def class_cat_bivariate(
        df: pd.DataFrame,
        flag: str,
        length: float,
        width: float,
        col_start: int,
        col_end: int,
    ) -> None:
        """
        Generate bar plots to visualize the percentage of a binary target variable across different categorical groups.

        Args:
            df (pd.DataFrame): DataFrame containing the data.
            flag (str): Column name of the binary target variable.
            length (float): Figure height for the plot.
            width (float): Figure width for the plot.
            col_start (int): Starting index for selecting categorical columns.
            col_end (int): Ending index for selecting categorical columns.

        Returns:
            None
        """
        X_cat = df.select_dtypes(include=["object"]).columns[col_start:col_end]
        for X in X_cat:
            label1 = round(df[[X, flag]].groupby([X]).sum(), 0)
            label2 = round(df[[X, flag]].groupby([X]).count(), 0)
            label3 = pd.concat([label1, label2], axis=1)
            label3.columns = ["sum", "count"]
            label3["rate"] = round((label3["sum"] / label3["count"]) * 100, 0)
            label3 = label3.sort_values(by=["rate"], ascending=True)
            label3["rate"].plot.barh(figsize=(width, length))
            plt.title("percentage " + flag + " per " + X)
            plt.xlabel("rate of " + flag)
            plt.ylabel(X)
            plt.show()
            label3 = label3.sort_values(by=["rate"], ascending=False)
            display(label3)

        return None

    @staticmethod
    def calculate_vif(
        X: pd.DataFrame, target: str, threshold: float, feature_elim: int
    ) -> Tuple[List[str], List[float]]:
        """
        Drop features one at a time until the VIF scores are below the given threshold.

        Args:
            X (pd.DataFrame): DataFrame containing the feature set (including the target column).
            target (str): Target variable to be used in the regression formula.
            threshold (float): VIF threshold above which features are dropped.
            feature_elim (int): Maximum number of features to eliminate.

        Returns:
            Tuple[List[str], List[float]]: A tuple containing a list of dropped feature names and their corresponding VIF scores.
        """
        feature_list: List[str] = []
        Feature_vif_list: List[float] = []
        max_iter = feature_elim
        iter_count = 0

        while iter_count <= max_iter:
            X_current = X.drop(feature_list, axis=1, errors="ignore")
            features = "+".join(X_current.columns[0 : len(X_current.columns) - 1])
            y, X1 = dmatrices(
                target + " ~" + features, X_current, return_type="dataframe"
            )
            vif = pd.DataFrame()
            vif["vif"] = [
                variance_inflation_factor(X1.values, i) for i in range(X1.shape[1])
            ]
            vif["features"] = X1.columns
            vif = vif.sort_values(by=["vif"], ascending=False).reset_index(drop=True)
            vif["vif2"] = vif["vif"]
            vif.loc[vif.features == "Intercept", "vif2"] = 0
            max_feature = vif.loc[vif["vif2"].idxmax()]
            max_feature_name = max_feature["features"]
            max_feature_vif = max_feature["vif"]
            if max_feature_vif > threshold and max_feature_name != "Intercept":
                feature_list.append(max_feature_name)
                Feature_vif_list.append(max_feature_vif)
                iter_count += 1
            else:
                iter_count += 1

        vif = vif.drop(["vif2"], axis=1, errors="ignore")
        display(vif)
        print("\ndropped features: ")

        return feature_list, Feature_vif_list

    @staticmethod
    def crosstabs(col_left: str, col_top: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a cross tabulation table for two given columns, compute chi-squared p-value,
        and format the table with percentages and additional totals.

        Args:
            col_left (str): Column name to be used as rows.
            col_top (str): Column name to be used as columns.
            df (pd.DataFrame): DataFrame containing the data.

        Returns:
            pd.DataFrame: A formatted crosstab DataFrame.
        """
        # Create cross tab
        crosstab = pd.crosstab(df[col_left], df[col_top])
        # Get p-value using chi-squared test
        try:
            chi2, p, dof, ex = chi2_contingency(crosstab, correction=True)
            p = round(p, 2)
        except Exception:
            p = 'no data; "observed" has size 0'
        # Allow for row totals
        crosstab = crosstab.apply(lambda r: r / r.sum(), axis=1)
        crosstab["Total"] = crosstab[crosstab.columns].sum(axis=1, numeric_only=True)
        # Convert to percentages
        crosstab = crosstab.multiply(100).astype(int)
        # Get counts totals
        crosstab_totals = pd.crosstab(df[col_left], df[col_top], margins=True)
        crosstab_totals = crosstab_totals[:-1]  # drop last row (margins)
        crosstab_totals = crosstab_totals[["All"]]
        crosstab_totals.columns = ["n Count"]
        crosstab = pd.concat([crosstab, crosstab_totals], axis=1)
        # Ensure total adds to 100
        crosstab["Total"] = np.where(
            (crosstab["Total"] < 100) & (crosstab["Total"] > 98), 100, crosstab["Total"]
        )
        # Apply esthetics and formatting
        crosstab["Feature"] = col_left
        crosstab = crosstab.reset_index()
        first_column = crosstab.pop("Feature")
        crosstab.insert(0, "Feature", first_column)
        crosstab.columns.values[1] = "Feature_Value"
        crosstab["Feature"] = crosstab["Feature"] + " - p-value: " + str(p)
        crosstab = crosstab.astype(str)
        crosstab.columns.name = None
        crosstab = crosstab.append(pd.Series(), ignore_index=True)
        crosstab = crosstab.fillna("")
        col_names = list(crosstab.columns[2:])
        col_names_list = [("", "Feature"), ("", "Feature_Value")]
        interation_count = 0
        for col in col_names:
            if interation_count == 0:
                col_tuple = (col_top, col)
                col_names_list.append(col_tuple)
                interation_count += 1
            else:
                col_tuple = ("", col)
                col_names_list.append(col_tuple)
                interation_count += 1
        crosstab.columns = pd.MultiIndex.from_tuples(col_names_list)
        crosstab["."] = ""

        return crosstab

    @staticmethod
    def crosstab_report(df: pd.DataFrame, drop_cols: List[str]) -> pd.DataFrame:
        """
        Generate a comprehensive crosstab report by creating crosstabs for each pair of columns
        (excluding specified columns).

        Args:
            df (pd.DataFrame): DataFrame containing the data.
            drop_cols (List[str]): List of column names to drop from the analysis.

        Returns:
            pd.DataFrame: A final concatenated DataFrame containing all crosstab reports.
        """
        freq_book_df = df.drop(drop_cols, axis=1, errors="ignore")
        final_crosstab_list = []
        # Loop through each column as the top value
        for col_top in list(freq_book_df.columns):
            crosstab_list = []
            # Loop through each column as the left value
            for col_left in list(freq_book_df.columns):
                ct = Preprocessing.crosstabs(
                    col_left=col_left, col_top=col_top, df=freq_book_df
                )
                crosstab_list.append(ct)
            final_crosstab = pd.concat(crosstab_list, axis=0)
            final_crosstab_list.append(final_crosstab)
        final_df = pd.concat(final_crosstab_list, axis=1).reset_index(drop=True)

        return final_df
