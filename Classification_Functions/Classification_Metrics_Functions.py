from typing import Any, List, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import *
from sklearn.metrics import auc, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score


class Metrics:
    @staticmethod
    def classification_cross_val_splits(
        classifier: Any,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        cv: Any,
        font: float,
    ) -> None:
        """
        Perform K-fold cross validation for classification metrics and plot results.

        Args:
            classifier (Any): Classifier model implementing the fit/predict interface.
            X_train (Union[pd.DataFrame, np.ndarray]): Training features.
            y_train (Union[pd.Series, np.ndarray]): Training target labels.
            cv (Any): Cross-validation splitting strategy (e.g., KFold).
            font (float): Font scaling factor for plots.

        Returns:
            None
        """
        # Accuracy
        accuracies_accuracy = cross_val_score(
            estimator=classifier,
            X=X_train,
            y=y_train,
            cv=cv,
            n_jobs=-1,
            scoring="accuracy",
        )
        accuracies_accuracy_mean = round(accuracies_accuracy.mean(), 4)
        accuracies_accuracy_std = round(accuracies_accuracy.std(), 4)

        # F1
        accuracies_f1 = cross_val_score(
            estimator=classifier, X=X_train, y=y_train, cv=cv, n_jobs=-1, scoring="f1"
        )
        accuracies_f1_mean = round(accuracies_f1.mean(), 4)
        accuracies_f1_std = round(accuracies_f1.std(), 4)

        # Precision
        accuracies_precision = cross_val_score(
            estimator=classifier,
            X=X_train,
            y=y_train,
            cv=cv,
            n_jobs=-1,
            scoring="precision",
        )
        accuracies_precision_mean = round(accuracies_precision.mean(), 4)
        accuracies_precision_std = round(accuracies_precision.std(), 4)

        # Recall
        accuracies_recall = cross_val_score(
            estimator=classifier,
            X=X_train,
            y=y_train,
            cv=cv,
            n_jobs=-1,
            scoring="recall",
        )
        accuracies_recall_mean = round(accuracies_recall.mean(), 4)
        accuracies_recall_std = round(accuracies_recall.std(), 4)

        print("\n" + "test model f-fold metrics:" + "\n")

        sns.set(font_scale=font, style="white")

        plt.figure(figsize=(10, 3))
        plt.plot(
            range(1, cv.get_n_splits(X_train) + 1, 1),
            accuracies_accuracy,
            ls="-",
            marker="o",
        )
        plt.title("accuracy for kfold")
        plt.xlabel("kfold index")
        plt.ylabel("accuracy")
        plt.ylim(0, 1.1)
        plt.show()

        results1 = [
            ("accuracy k-fold cross validation: ", accuracies_accuracy_mean),
            ("accuracy std: ", accuracies_accuracy_std),
        ]
        for label, value in results1:
            print(f"{label:{50}} {value:.>{20}}")

        plt.figure(figsize=(10, 3))
        plt.plot(
            range(1, cv.get_n_splits(X_train) + 1, 1), accuracies_f1, ls="-", marker="o"
        )
        plt.title("f1 for kfold")
        plt.xlabel("kfold index")
        plt.ylabel("f1")
        plt.ylim(0, max(accuracies_f1) * 1.25)
        plt.show()

        results2 = [
            ("f1 k-fold cross validation: ", accuracies_f1_mean),
            ("f1 std: ", accuracies_f1_std),
        ]
        print("\n")
        for label, value in results2:
            print(f"{label:{50}} {value:.>{20}}")

        plt.figure(figsize=(10, 3))
        plt.plot(
            range(1, cv.get_n_splits(X_train) + 1, 1),
            accuracies_precision,
            ls="-",
            marker="o",
        )
        plt.title("precision for kfold")
        plt.xlabel("kfold index")
        plt.ylabel("precision")
        plt.ylim(0, max(accuracies_precision) * 1.25)
        plt.show()

        results3 = [
            ("precision k-fold cross validation: ", accuracies_precision_mean),
            ("precision std: ", accuracies_precision_std),
        ]
        print("\n")
        for label, value in results3:
            print(f"{label:{50}} {value:.>{20}}")

        plt.figure(figsize=(10, 3))
        plt.plot(
            range(1, cv.get_n_splits(X_train) + 1, 1),
            accuracies_recall,
            ls="-",
            marker="o",
        )
        plt.title("recall for kfold")
        plt.xlabel("kfold index")
        plt.ylabel("recall")
        plt.ylim(0, max(accuracies_recall) * 1.25)
        plt.show()

        results4 = [
            ("recall k-fold cross validation: ", accuracies_recall_mean),
            ("recall std: ", accuracies_recall_std),
        ]
        print("\n")
        for label, value in results4:
            print(f"{label:{50}} {value:.>{20}}")
        print("\n")

        return None

    @staticmethod
    def cap_auc(
        model: Any,
        df: pd.DataFrame,
        target: str,
        y: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray],
        y_score: Union[pd.Series, np.ndarray],
        X: Union[pd.DataFrame, np.ndarray],
        length: float,
        width: float,
        ks: float,
        text: str,
        font: float,
    ) -> pd.DataFrame:
        """
        Generates classification metrics including concordance/discordance, ROC plot, confusion matrix,
        cumulative accuracy profile (CAP), and CAP table.

        Args:
            model (Any): Classification model with a predict_proba method.
            df (pd.DataFrame): DataFrame containing actual target values and predicted probabilities.
            target (str): Target column name.
            y (Union[pd.Series, np.ndarray]): Actual target values.
            y_pred (Union[pd.Series, np.ndarray]): Predicted class labels.
            y_score (Union[pd.Series, np.ndarray]): Predicted probabilities for positive class.
            X (Union[pd.DataFrame, np.ndarray]): Feature set used for prediction.
            length (float): Height of the figure for plots.
            width (float): Width of the figure for plots.
            ks (float): KS threshold percentage value.
            text (str): Text annotation for the CAP plot.
            font (float): Font scaling factor for plots.

        Returns:
            pd.DataFrame: DataFrame containing CAP table thresholds.
        """
        Probability = model.predict_proba(X)
        Probability1 = pd.DataFrame(Probability)
        Probability1.columns = ["Prob_0", "Prob_1"]

        # Merge target values with predicted probabilities
        TruthTable = pd.merge(
            y[[target]],
            Probability1[["Prob_1"]],
            how="inner",
            left_index=True,
            right_index=True,
        )

        zeros = TruthTable[TruthTable[target] == 0].reset_index(drop=True)
        ones = TruthTable[TruthTable[target] == 1].reset_index(drop=True)

        from bisect import bisect_left, bisect_right

        zeros_list = sorted([zeros.iloc[j, 1] for j in zeros.index])
        zeros_length = len(zeros_list)
        disc = 0
        ties = 0
        conc = 0

        for i in ones.index:
            cur_conc = bisect_left(zeros_list, ones.iloc[i, 1])
            cur_ties = bisect_right(zeros_list, ones.iloc[i, 1]) - cur_conc
            conc += cur_conc
            ties += cur_ties

        pairs_tested = zeros_length * len(ones.index)
        disc = pairs_tested - conc - ties
        concordance = round(conc / pairs_tested, 2)
        discordance = round(disc / pairs_tested, 2)
        ties_perc = round(ties / pairs_tested, 2)
        Somers_D = round((conc - disc) / pairs_tested, 2)
        auc_score = round(roc_auc_score(y, y_score), 2)

        results1 = [
            ("Pairs: ", pairs_tested),
            ("Conc: ", conc),
            ("Disc: ", disc),
            ("Tied: ", ties),
        ]
        print("\n")
        for label, value in results1:
            print(f"{label:{35}} {value:.>{20}}")

        results2 = [
            ("Concordance: ", concordance),
            ("Discordance: ", discordance),
            ("Tied: ", ties_perc),
        ]
        print("\n")
        for label, value in results2:
            print(f"{label:{35}} {value:.>{20}}")

        results = [("AUC:", auc_score)]
        print("\n")
        for label, value in results:
            print(f"{label:{35}} {value:.>{20}}")

        # ROC plot
        probs = y_score
        fpr, tpr, thresholds = roc_curve(y, probs)
        roc_auc = auc(fpr, tpr)
        print("\n")
        plt.figure(figsize=(width, length))
        plt.plot([0, 1], [0, 1], "r--", label="random model")
        label_str = "classifier:" + " {0:.2f}".format(roc_auc)
        plt.plot(fpr, tpr, c="g", label=label_str, linewidth=2)
        plt.xlabel("false positive rate")
        plt.ylabel("true positive rate")
        plt.title("receiver operating characteristic")
        plt.legend(loc="lower right")
        plt.show()

        # General Classification metrics
        cm = confusion_matrix(y, y_pred)
        tn = cm[0][0]
        fp = cm[0][1]
        tp = cm[1][1]
        fn = cm[1][0]

        print("\n")
        cm_df = pd.DataFrame([{"1": tp, "0": fp}, {"1": fn, "0": tn}])
        cm_df = cm_df.set_index(pd.Index([1, 0]))
        print("confussion matrix:" + "\n")
        display(cm_df)

        accuracy = round((tp + tn) / (tp + fp + tn + fn), 2)
        precision = round(tp / (tp + fp), 2)
        recall = round(tp / (tp + fn), 2)
        f1 = round((2 * (precision * recall)) / (precision + recall), 2)
        false_positive_rate = round(fp / (fp + tn), 2)
        true_positive_rate = round(tp / (tp + fn), 2)
        percent_of_increase_per_unit = round(
            true_positive_rate / false_positive_rate, 2
        )

        results3 = [
            ("Accuracy:", accuracy),
            ("Precision (tp / (tp + fp)):", precision),
            ("Recall (tp / (tp + fn)):", recall),
            ("F1 ((2 * (precision * recall)) / (precision + recall)):", f1),
            ("False Positive Rate (fp / (fp + tn))", false_positive_rate),
            ("True Positive Rate (tp / (tp + fn)):", true_positive_rate),
            (
                "Increase in TPR / FPR (true_positive_rate / false_positive_rate):",
                percent_of_increase_per_unit,
            ),
        ]
        print("\n")
        for label, value in results3:
            print(f"{label:{75}} {value:.>{20}}")

        # CAP plot
        y_array = y.to_numpy() if isinstance(y, pd.Series) else y
        y_pred_array = (
            y_pred.astype(int).to_numpy()
            if isinstance(y_pred, pd.Series)
            else y_pred.astype(int)
        )
        y_score_array = (
            y_score.to_numpy() if isinstance(y_score, pd.Series) else y_score
        )

        total = len(y_array)
        class_1_count = np.sum(y_array)
        class_0_count = total - class_1_count

        probs = y_score_array
        model_y = [y_val for _, y_val in sorted(zip(probs, y_array), reverse=True)]
        y_values = np.append([0], np.cumsum(model_y))
        X_values = np.arange(0, total + 1)

        print("\n")
        sns.set(font_scale=font, style="white")
        plt.figure(figsize=(width, length))
        plt.plot(
            [0, total], [0, class_1_count], c="r", linestyle="--", label="random model"
        )
        plt.plot(
            [0, class_1_count, total],
            [0, class_1_count, class_1_count],
            c="grey",
            linewidth=2,
            label="perfect model",
        )
        plt.plot(X_values, y_values, c="b", label="classifier", linewidth=2)

        index = int((ks * total / 100))
        plt.plot(
            [index, index], [0, y_values[index]], c="g", linestyle="--", label="ks"
        )
        plt.plot([0, index], [y_values[index], y_values[index]], c="g", linestyle="--")

        plt.xlabel("total observations")
        plt.ylabel("class 1 observations")
        plt.title("cumulative accuracy profile")
        plt.legend(loc="lower right")
        plt.text(index * 1.05, y_values[index] * 0.70, text)
        plt.show()

        # CAP table
        rows_decile = round(len(df) / 10, 0)
        flag_count = df[target].sum()
        cap_table = df.sort_values(by="predicted_proba", ascending=False).reset_index(
            drop=True
        )
        cap_table["count"] = 1
        cap_table["count_of_rows"] = 1
        cap_table["count"] = cap_table["count"].cumsum()
        cap_table["bin"] = np.ceil(cap_table["count"] / rows_decile)
        cap_table["bin"][cap_table["bin"] > 10] = 10

        # Get thresholds per bin (min, max, mean, median)
        cap_table_threshold_min = (
            cap_table.groupby(by=["bin"])["predicted_proba"]
            .min()
            .reset_index()
            .rename(columns={"predicted_proba": "min_predicted_proba_per_bin"})
        )
        cap_table_threshold_max = (
            cap_table.groupby(by=["bin"])["predicted_proba"]
            .max()
            .reset_index()
            .rename(columns={"predicted_proba": "max_predicted_proba_per_bin"})
        )
        cap_table_threshold_mean = (
            cap_table.groupby(by=["bin"])["predicted_proba"]
            .mean()
            .reset_index()
            .rename(columns={"predicted_proba": "mean_predicted_proba_per_bin"})
        )
        cap_table_threshold_median = (
            cap_table.groupby(by=["bin"])["predicted_proba"]
            .median()
            .reset_index()
            .rename(columns={"predicted_proba": "median_predicted_proba_per_bin"})
        )
        cap_table_threshold = cap_table_threshold_min.merge(
            cap_table_threshold_max, on="bin", how="left"
        )
        cap_table_threshold = cap_table_threshold.merge(
            cap_table_threshold_mean, on="bin", how="left"
        )
        cap_table_threshold = cap_table_threshold.merge(
            cap_table_threshold_median, on="bin", how="left"
        )

        cap_table = cap_table.groupby(by=["bin"]).sum().reset_index()
        cap_table = cap_table[["bin", "count_of_rows", target]]
        cap_table["model_percent"] = round((cap_table[target] / flag_count) * 100, 2)
        cap_table["random_percent"] = 10
        cap_table["model_cumm_percent"] = cap_table["model_percent"].cumsum()
        cap_table["random_cumm_percent"] = cap_table["random_percent"].cumsum()
        cap_table["ks"] = round(
            cap_table["model_cumm_percent"] - cap_table["random_cumm_percent"], 2
        )
        cap_table.loc[len(cap_table)] = 0
        cap_table = cap_table.sort_values(by="bin", ascending=True).reset_index(
            drop=True
        )

        print("\n")
        display(cap_table)
        print("\n")

        return cap_table_threshold

    @staticmethod
    def logit_summary(
        X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]
    ) -> Any:
        """
        Provides a summary of the logistic regression model.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Feature set.
            y (Union[pd.Series, np.ndarray]): Target variable.

        Returns:
            Any: Summary of the logistic regression model.
        """
        X2 = sm.add_constant(X)
        logit_model = sm.Logit(y, X2)
        result = logit_model.fit()

        return result.summary2()

    @staticmethod
    def classification_feature_importance(
        model: Any,
        X_cols: pd.DataFrame,
        font: float,
        length: float,
        width: float,
        pos: str,
        neg: str,
    ) -> None:
        """
        Plots feature importance based on model coefficients and displays the coefficient values.

        Args:
            model (Any): Trained classification model with 'coef_' and 'intercept_' attributes.
            X_cols (pd.DataFrame): DataFrame containing feature columns.
            font (float): Font scaling factor for plots.
            length (float): Height of the figure.
            width (float): Width of the figure.
            pos (str): Color for positive coefficients.
            neg (str): Color for negative coefficients.

        Returns:
            None
        """
        coefficients = model.coef_
        intercept = np.array([model.intercept_])
        coefficients = np.concatenate([coefficients, intercept], axis=1)
        coefficients = coefficients.reshape((-1, 1))
        X_col = np.array(X_cols.columns)
        X_col = np.concatenate([X_col, np.array(["intercept"])], axis=0)
        X_col = X_col.reshape((-1, 1))
        coefficients = np.concatenate((X_col, coefficients), axis=1)
        coefficients = pd.DataFrame(coefficients, columns=["features", "coef"])
        coefficients["positive"] = coefficients["coef"] > 0
        coefficients["coef2"] = abs(coefficients["coef"])
        coefficients = coefficients.sort_values(by=["coef2"], ascending=True)
        coefficients = coefficients.reset_index(drop=True)
        blue_patch = mpatches.Patch(color="#1f77b4", label="positive")
        red_patch = mpatches.Patch(color="r", label="negative")

        sns.set(font_scale=font, style="white")
        coefficients.plot(
            x="features",
            y="coef",
            kind="barh",
            figsize=(width, length),
            color=coefficients.positive.map({True: pos, False: neg}),
        )
        plt.title("feature importance (features scaled)")
        plt.xlabel("coefficient units")
        plt.ylabel("features")
        plt.legend(
            handles=[blue_patch, red_patch],
            bbox_to_anchor=(1.05, 1.0),
            loc="upper left",
        )
        plt.show()

        coefficients = coefficients.sort_values(by=["coef2"], ascending=False)
        coefficients = coefficients.drop(["coef2"], axis=1, errors="ignore")
        display(coefficients)

    @staticmethod
    def ks_features(
        df: pd.DataFrame,
        predicted_proba: str,
        drop_cols: List[str],
        ks_proba: float,
        length: float,
        width: float,
        font: float,
    ) -> None:
        """
        Analyzes feature proportions for subsets of data based on a KS probability threshold and plots the results.

        Args:
            df (pd.DataFrame): DataFrame containing features and predicted probabilities.
            predicted_proba (str): Column name for predicted probabilities.
            drop_cols (List[str]): List of columns to drop from the analysis.
            ks_proba (float): Threshold value for KS analysis.
            length (float): Height of the plot figure.
            width (float): Width of the plot figure.
            font (float): Font scaling factor for plots.

        Returns:
            None
        """
        ks_df = df.loc[df[predicted_proba] >= ks_proba].reset_index(drop=True)
        ks_df = ks_df.drop(drop_cols, axis=1, errors="ignore")

        # Get proportions per feature for ks subset
        ks_df_sum = ks_df.sum(axis=0) / ks_df.shape[0]
        ks_df_sum = pd.DataFrame(
            {"features": ks_df_sum.index, "KS_feature_proportions": ks_df_sum.values}
        )
        ks_df_sum = ks_df_sum.iloc[::-1]

        # For data below the threshold
        ks_diff_df = df.loc[df[predicted_proba] < ks_proba].reset_index(drop=True)
        ks_diff_df = ks_diff_df.drop(drop_cols, axis=1, errors="ignore")
        ks_diff_df_sum = ks_diff_df.sum(axis=0) / ks_diff_df.shape[0]
        ks_diff_df_sum = pd.DataFrame(
            {
                "features": ks_diff_df_sum.index,
                "KS_diff_feature_proportions": ks_diff_df_sum.values,
            }
        )
        ks_diff_df_sum = ks_diff_df_sum.iloc[::-1]

        # Merge the results
        ks_df = ks_df_sum.merge(ks_diff_df_sum, on="features", how="left")
        y_cols = ["KS_feature_proportions", "KS_diff_feature_proportions"]

        sns.set(font_scale=font, style="white")
        ks_df.plot(
            x="features", y=y_cols, kind="barh", figsize=(length, width), width=0.8
        )
        plt.title("Feature Frequencies(%) Per Model Results")
        plt.xlabel("Percentages")
        plt.ylabel("Features")
        print("\n")
        plt.show()

        display(ks_df)

        return None

    @staticmethod
    def boostrap_f1(
        cap_threshold_df: pd.DataFrame,
        x_axis: str,
        results_df: pd.DataFrame,
        real_results_col: str,
        pred_col: str,
        pred_proba_col: str,
        length: float,
        width: float,
        font: float,
    ) -> None:
        """
        Bootstraps f1 scores by adjusting prediction thresholds and plots f1 scores across bins.

        Args:
            cap_threshold_df (pd.DataFrame): DataFrame containing threshold metrics per bin.
            x_axis (str): Column name to be used as the x-axis in the plot.
            results_df (pd.DataFrame): DataFrame containing real and predicted results.
            real_results_col (str): Column name for actual results.
            pred_col (str): Column name for predicted labels.
            pred_proba_col (str): Column name for predicted probabilities.
            length (float): Height of the plot figure.
            width (float): Width of the plot figure.
            font (float): Font scaling factor for plots.

        Returns:
            None
        """
        bin_list = cap_threshold_df["bin"].tolist()
        threshold_min_list = cap_threshold_df["min_predicted_proba_per_bin"].tolist()
        threshold_max_list = cap_threshold_df["max_predicted_proba_per_bin"].tolist()
        threshold_mean_list = cap_threshold_df["mean_predicted_proba_per_bin"].tolist()
        threshold_median_list = cap_threshold_df[
            "median_predicted_proba_per_bin"
        ].tolist()

        # f1 scores
        f1_min = []
        f1_max = []
        f1_mean = []
        f1_median = []

        # Thresholds
        threshold_min = []
        threshold_max = []
        threshold_mean = []
        threshold_median = []

        for min_, max_, mean, median in zip(
            threshold_min_list,
            threshold_max_list,
            threshold_mean_list,
            threshold_median_list,
        ):
            # Adjust prediction threshold min
            results_df[pred_col] = np.where(results_df[pred_proba_col] >= min_, 1, 0)
            cm = confusion_matrix(
                results_df[[real_results_col]], results_df[[pred_col]]
            )
            tn, fp = cm[0][0], cm[0][1]
            tp, fn = cm[1][1], cm[1][0]
            precision_val = round(tp / (tp + fp), 2)
            recall_val = round(tp / (tp + fn), 2)
            f1_val = round(
                (2 * (precision_val * recall_val)) / (precision_val + recall_val), 2
            )
            f1_min.append(f1_val)
            threshold_min.append(min_)

            # Adjust prediction threshold max
            results_df[pred_col] = np.where(results_df[pred_proba_col] >= max_, 1, 0)
            cm = confusion_matrix(
                results_df[[real_results_col]], results_df[[pred_col]]
            )
            tn, fp = cm[0][0], cm[0][1]
            tp, fn = cm[1][1], cm[1][0]
            precision_val = round(tp / (tp + fp), 2)
            recall_val = round(tp / (tp + fn), 2)
            f1_val = round(
                (2 * (precision_val * recall_val)) / (precision_val + recall_val), 2
            )
            f1_max.append(f1_val)
            threshold_max.append(max_)

            # Adjust prediction threshold mean
            results_df[pred_col] = np.where(results_df[pred_proba_col] >= mean, 1, 0)
            cm = confusion_matrix(
                results_df[[real_results_col]], results_df[[pred_col]]
            )
            tn, fp = cm[0][0], cm[0][1]
            tp, fn = cm[1][1], cm[1][0]
            precision_val = round(tp / (tp + fp), 2)
            recall_val = round(tp / (tp + fn), 2)
            f1_val = round(
                (2 * (precision_val * recall_val)) / (precision_val + recall_val), 2
            )
            f1_mean.append(f1_val)
            threshold_mean.append(mean)

            # Adjust prediction threshold median
            results_df[pred_col] = np.where(results_df[pred_proba_col] >= median, 1, 0)
            cm = confusion_matrix(
                results_df[[real_results_col]], results_df[[pred_col]]
            )
            tn, fp = cm[0][0], cm[0][1]
            tp, fn = cm[1][1], cm[1][0]
            precision_val = round(tp / (tp + fp), 2)
            recall_val = round(tp / (tp + fn), 2)
            f1_val = round(
                (2 * (precision_val * recall_val)) / (precision_val + recall_val), 2
            )
            f1_median.append(f1_val)
            threshold_median.append(median)

        plot_data = pd.DataFrame(
            {
                "bin": bin_list,
                "f1_min": f1_min,
                "f1_max": f1_max,
                "f1_mean": f1_mean,
                "f1_median": f1_median,
            }
        )

        read_data = pd.DataFrame(
            {
                "bin": bin_list,
                "threshold_min": threshold_min,
                "f1_min": f1_min,
                "threshold_max": threshold_max,
                "f1_max": f1_max,
                "threshold_mean": threshold_mean,
                "f1_mean": f1_mean,
                "theshold_median": threshold_median,
                "f1_median": f1_median,
            }
        )

        sns.set(font_scale=font, style="white")
        plot_data.plot.line(x=x_axis, figsize=(width, length))
        plt.title("f1 scores per bin (decile) grouping by different metrics")
        plt.xlabel("bins")
        plt.ylabel("f1 scores")
        print("\n")
        plt.show()

        print("\n")
        display(read_data)
        print("\n")

        return None

    @staticmethod
    def generate_cap_curve_and_table(
        df: pd.DataFrame,
        target: str,
        y: pd.Series,
        y_score: pd.Series,
        ks: int,
        text: str,
        font: float,
        width: int,
        length: int,
    ) -> pd.DataFrame:
        """
        Generates a Cumulative Accuracy Profile (CAP) curve and calculates the CAP table.

        Parameters:
        -----------
        df : pd.DataFrame
            The dataset containing actual target values and predicted probabilities.
        target : str
            The column name of the actual target variable.
        y : pd.Series
            The actual target values.
        y_score : pd.Series
            The predicted probabilities.
        ks : int, optional (default=40)
            The KS statistic threshold percentage.
        text : str, optional (default="KS: 40")
            Text annotation for the KS statistic.
        font : float, optional (default=1)
            Font scaling factor for the plot.
        width : int, optional (default=10)
            Width of the plot.
        length : int, optional (default=5)
            Height of the plot.

        Returns:
        --------
        pd.DataFrame
            The CAP table containing binned statistics.
        """

        # Convert to NumPy arrays if needed
        y_array = y.to_numpy() if isinstance(y, pd.Series) else np.array(y)
        y_score_array = (
            y_score.to_numpy() if isinstance(y_score, pd.Series) else np.array(y_score)
        )

        # CAP Plot Data Calculation
        total = len(y_array)
        class_1_count = np.sum(y_array)

        probs = y_score_array
        model_y = [y_val for _, y_val in sorted(zip(probs, y_array), reverse=True)]
        y_values = np.append([0], np.cumsum(model_y))
        X_values = np.arange(0, total + 1)

        # Plot CAP Curve
        sns.set(font_scale=float(font), style="white")
        plt.figure(figsize=(width, length))
        plt.plot(
            [0, total], [0, class_1_count], c="r", linestyle="--", label="random model"
        )
        plt.plot(
            [0, class_1_count, total],
            [0, class_1_count, class_1_count],
            c="grey",
            linewidth=2,
            label="perfect model",
        )
        plt.plot(X_values, y_values, c="b", label="classifier", linewidth=2)

        # KS Statistic Line
        index = int((ks * total / 100))
        plt.plot(
            [index, index], [0, y_values[index]], c="g", linestyle="--", label="ks"
        )
        plt.plot([0, index], [y_values[index], y_values[index]], c="g", linestyle="--")

        # Labels and legend
        plt.xlabel("Total Observations")
        plt.ylabel("Class 1 Observations")
        plt.title("Cumulative Accuracy Profile")
        plt.legend(loc="lower right")
        plt.text(index * 1.05, y_values[index] * 0.70, text)

        plt.show()

        # CAP Table Calculation
        rows_decile = round(len(df) / 10, 0)
        flag_count = df[target].sum()

        cap_table = df.sort_values(by="predicted_proba", ascending=False).reset_index(
            drop=True
        )

        # Assign cumulative count
        cap_table["count"] = np.arange(1, len(cap_table) + 1)

        # Assign bins correctly using .loc[]
        cap_table["bin"] = np.ceil(cap_table["count"] / rows_decile).astype(int)

        # Clip values to ensure bins do not exceed 10
        cap_table["bin"] = cap_table["bin"].clip(upper=10)

        # Aggregate CAP table
        cap_table = (
            cap_table.groupby("bin")
            .agg(count_of_rows=("count", "count"), total_target=(target, "sum"))
            .reset_index()
        )

        cap_table["model_percent"] = round(
            (cap_table["total_target"] / flag_count) * 100, 2
        )
        cap_table["random_percent"] = 10
        cap_table["model_cumm_percent"] = cap_table["model_percent"].cumsum()
        cap_table["random_cumm_percent"] = cap_table["random_percent"].cumsum()
        cap_table["ks"] = round(
            cap_table["model_cumm_percent"] - cap_table["random_cumm_percent"], 2
        )

        # Ensure last row is included with 0 values
        cap_table.loc[len(cap_table)] = 0
        cap_table = cap_table.sort_values(by="bin", ascending=True).reset_index(
            drop=True
        )

        return cap_table
