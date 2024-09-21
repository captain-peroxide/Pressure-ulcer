import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.calibration import calibration_curve
import numpy as np

class MLPlots:
    def __init__(self):
        sns.set(style="whitegrid")

    def plot_confusion_matrix(self, y_true, y_pred, labels=None, normalize=False, title='Confusion Matrix'):
        cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true' if normalize else None)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

    def plot_roc_curve(self, y_true, y_proba, title='ROC Curve'):
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(10, 7))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.show()

    def plot_precision_recall_curve(self, y_true, y_proba, title='Precision-Recall Curve'):
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        plt.figure(figsize=(10, 7))
        plt.plot(recall, precision, color='b', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.show()

    def plot_feature_importance(self, model, feature_names, title='Feature Importance'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(10, 7))
        plt.title(title)
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.xlim([-1, len(importances)])
        plt.show()

    def plot_learning_curve(self, train_sizes, train_scores, test_scores, title='Learning Curve'):
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.figure(figsize=(10, 7))
        plt.title(title)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")

        plt.legend(loc="best")
        plt.show()

    def plot_residuals(self, y_true, y_pred, title='Residuals Plot'):
        residuals = y_true - y_pred
        plt.figure(figsize=(10, 7))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.hlines(y=0, xmin=min(y_pred), xmax=max(y_pred), colors='r', linestyles='dashed')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title(title)
        plt.show()

    def plot_calibration_curve(self, y_true, y_proba, title='Calibration Curve'):
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
        plt.figure(figsize=(10, 7))
        plt.plot(prob_pred, prob_true, marker='o', label='Calibration curve')
        plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
        plt.xlabel('Mean predicted probability')
        plt.ylabel('Fraction of positives')
        plt.title(title)
        plt.legend(loc="best")
        plt.show()

    def plot_cumulative_gain(self, y_true, y_proba, title='Cumulative Gain Chart'):
        sorted_indices = np.argsort(y_proba)[::-1]
        sorted_true = np.array(y_true)[sorted_indices]
        cumulative_gains = np.cumsum(sorted_true) / np.sum(sorted_true)
        plt.figure(figsize=(10, 7))
        plt.plot(np.arange(1, len(cumulative_gains) + 1) / len(cumulative_gains), cumulative_gains, marker='o')
        plt.plot([0, 1], [0, 1], linestyle='--', label='Baseline')
        plt.xlabel('Proportion of samples')
        plt.ylabel('Cumulative gain')
        plt.title(title)
        plt.legend(loc="best")
        plt.show()

    def plot_lift_curve(self, y_true, y_proba, title='Lift Curve'):
        sorted_indices = np.argsort(y_proba)[::-1]
        sorted_true = np.array(y_true)[sorted_indices]
        cumulative_gains = np.cumsum(sorted_true) / np.sum(sorted_true)
        lift = cumulative_gains / (np.arange(1, len(cumulative_gains) + 1) / len(cumulative_gains))
        plt.figure(figsize=(10, 7))
        plt.plot(np.arange(1, len(lift) + 1) / len(lift), lift, marker='o')
        plt.xlabel('Proportion of samples')
        plt.ylabel('Lift')
        plt.title(title)
        plt.show()


    def plot_validation_curve(self, param_range, train_scores, test_scores, title='Validation Curve'):
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.figure(figsize=(10, 7))
        plt.title(title)
        plt.xlabel("Parameter value")
        plt.ylabel("Score")
        plt.grid()

        plt.fill_between(param_range, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(param_range, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(param_range, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(param_range, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")

        plt.legend(loc="best")
        plt.show()

