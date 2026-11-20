"""Point metrics, offline and online: Counts of TP, TN, FP, FN, Precision, Recall, F1, AUC-ROC,
AUC-PR."""

import numpy as np

from tsbenchmark.evaluation_metrics._utils import area_under_curve, curve, plot_curve


def tp(anm_labels_test, anm_labels_pred):
    return np.sum(anm_labels_test * anm_labels_pred)


def tn(anm_labels_test, anm_labels_pred):
    return np.sum((1 - anm_labels_test) * (1 - anm_labels_pred))


def fp(anm_labels_test, anm_labels_pred):
    return np.sum((1 - anm_labels_test) * anm_labels_pred)


def fn(anm_labels_test, anm_labels_pred):
    return np.sum(anm_labels_test * (1 - anm_labels_pred))


def precision(anm_labels_test, anm_labels_pred):
    num_alarms = np.sum(anm_labels_pred)
    return tp(anm_labels_test, anm_labels_pred) / num_alarms if num_alarms > 0 else np.nan


def recall(anm_labels_test, anm_labels_pred):
    num_anomalies = np.sum(anm_labels_test)
    return tp(anm_labels_test, anm_labels_pred) / num_anomalies if num_anomalies > 0 else np.nan


def f1(anm_labels_test, anm_labels_pred):
    total = np.sum(anm_labels_test) + np.sum(anm_labels_pred)
    return 2 * tp(anm_labels_test, anm_labels_pred) / total if total > 0 else np.nan


def roc_point(anm_labels_test, anm_labels_pred):
    num_normals = np.sum(1 - anm_labels_test)
    fpr = fp(anm_labels_test, anm_labels_pred) / num_normals if num_normals > 0 else np.nan
    return fpr, recall(anm_labels_test, anm_labels_pred)


def pr_point(anm_labels_test, anm_labels_pred):
    p = precision(anm_labels_test, anm_labels_pred)
    return recall(anm_labels_test, anm_labels_pred), 1.0 if np.isnan(p) else p  # no alarm: 1


def auc_roc(anm_labels_test, anm_scores_pred, plot=False):
    fprs, tprs = curve(roc_point, anm_labels_test, anm_scores_pred)
    auc = area_under_curve(fprs, tprs)
    if plot:
        plot_curve(fprs, tprs, "false positive rate", "true positive rate", f"AUC-ROC = {auc:.3f}")
    return auc


def auc_pr(anm_labels_test, anm_scores_pred, plot=False):
    recalls, precisions = curve(pr_point, anm_labels_test, anm_scores_pred)
    auc = area_under_curve(recalls, precisions)
    if plot:
        plot_curve(recalls, precisions, "recall", "precision", f"AUC-PR = {auc:.3f}")
    return auc
