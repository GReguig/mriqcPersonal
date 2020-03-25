import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import islice
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import LabelEncoder
from os.path import join as opj
from copy import copy


def read_dataset(fpaths, drop_site=False):
    x_path, y_path = fpaths[0], fpaths[1]
    x_df, y_df = pd.read_csv(x_path), pd.read_csv(y_path)
    merged = x_df.merge(y_df, how="inner", on="subject_id")
    y_values = merged[["y", "subject_id"]]

    x_values = merged.drop(columns=["y"])
    if drop_site and "site" in x_values.columns:
        x_values = x_values.drop(columns=["site"])
    else:
        cols = list(x_values.columns)
        cols.remove("site")
        cols = cols + ["site"]
        x_values = x_values[cols]
    x_values = x_values.set_index(keys="subject_id")
    y_values = y_values.set_index(keys="subject_id")

    return {"x": x_values, "y": y_values}


def read_all_datasets_mriqc(drop_site=False):
    paths = {
            "abide": ["/network/lustre/iss01/cenir/analyse/irm/users/ghiles.reguig/These/Dataset/mriqc/ABIDE/x_abide_clean.csv",
                      "/network/lustre/iss01/cenir/analyse/irm/users/ghiles.reguig/These/Dataset/mriqc/ABIDE/y.csv"],

            "ds30": ["/network/lustre/iss01/cenir/analyse/irm/users/ghiles.reguig/These/Dataset/mriqc/ds30/x_ds030.csv",
                     "/network/lustre/iss01/cenir/analyse/irm/users/ghiles.reguig/These/Dataset/mriqc/ds30/y.csv"],

            "ds30_nogh": ["/network/lustre/iss01/cenir/analyse/irm/users/ghiles.reguig/These/Dataset/mriqc/ds30/x_ds30_noghost.csv",
                          "/network/lustre/iss01/cenir/analyse/irm/users/ghiles.reguig/These/Dataset/mriqc/ds30/y_noghost.csv"],

            "cati0": ["/network/lustre/iss01/cenir/analyse/irm/users/ghiles.reguig/These/Dataset/mriqc/CATI/cati_mriqc_classif.csv",
                      "/network/lustre/iss01/cenir/analyse/irm/users/ghiles.reguig/These/Dataset/mriqc/CATI/l0vs1234_y.csv"],

            "cati1": [
                "/network/lustre/iss01/cenir/analyse/irm/users/ghiles.reguig/These/Dataset/mriqc/CATI/cati_mriqc_classif.csv",
                "/network/lustre/iss01/cenir/analyse/irm/users/ghiles.reguig/These/Dataset/mriqc/CATI/l01vs234_y.csv"],

            "cati2": [
                "/network/lustre/iss01/cenir/analyse/irm/users/ghiles.reguig/These/Dataset/mriqc/CATI/cati_mriqc_classif.csv",
                "/network/lustre/iss01/cenir/analyse/irm/users/ghiles.reguig/These/Dataset/mriqc/CATI/l012vs34_y.csv"],

            "cati3": [
                "/network/lustre/iss01/cenir/analyse/irm/users/ghiles.reguig/These/Dataset/mriqc/CATI/cati_mriqc_classif.csv",
                "/network/lustre/iss01/cenir/analyse/irm/users/ghiles.reguig/These/Dataset/mriqc/CATI/l0123vs4_y.csv"],
            }
    
    """        
    'cati0_full': [
        '/network/lustre/iss01/cenir/analyse/irm/users/ghiles.reguig/These/Dataset/mriqc/CATI_full/x.csv',
        '/network/lustre/iss01/cenir/analyse/irm/users/ghiles.reguig/These/Dataset/mriqc/CATI_full/0vs1234.csv'],
            
    'cati1_full': [
        '/network/lustre/iss01/cenir/analyse/irm/users/ghiles.reguig/These/Dataset/mriqc/CATI_full/x.csv',
        '/network/lustre/iss01/cenir/analyse/irm/users/ghiles.reguig/These/Dataset/mriqc/CATI_full/01vs234.csv'],
            
    'cati2_full': [
        '/network/lustre/iss01/cenir/analyse/irm/users/ghiles.reguig/These/Dataset/mriqc/CATI_full/x.csv',
        '/network/lustre/iss01/cenir/analyse/irm/users/ghiles.reguig/These/Dataset/mriqc/CATI_full/012vs34.csv'],
    
    'cati3_full': [
        '/network/lustre/iss01/cenir/analyse/irm/users/ghiles.reguig/These/Dataset/mriqc/CATI_full/x.csv',
        '/network/lustre/iss01/cenir/analyse/irm/users/ghiles.reguig/These/Dataset/mriqc/CATI_full/0123vs4.csv'],
    """

    datasets = {ds_name: read_dataset(ds_paths, drop_site=drop_site) for ds_name, ds_paths in paths.items()}

    return datasets


def read_all_datasets_cat12(drop_site=False):
    paths = {
        "abide": [
            "/network/lustre/iss01/cenir/analyse/irm/users/ghiles.reguig/These/Dataset/cat12/ABIDE/x_abide.csv",
            "/network/lustre/iss01/cenir/analyse/irm/users/ghiles.reguig/These/Dataset/cat12/ABIDE/y.csv"],

        "ds30": ["/network/lustre/iss01/cenir/analyse/irm/users/ghiles.reguig/These/Dataset/cat12/ds30/x_ds030.csv",
                 "/network/lustre/iss01/cenir/analyse/irm/users/ghiles.reguig/These/Dataset/cat12/ds30/y.csv"],

        "cati0": [
            "/network/lustre/iss01/cenir/analyse/irm/users/ghiles.reguig/These/Dataset/cat12/CATI/x_cati.csv",
            "/network/lustre/iss01/cenir/analyse/irm/users/ghiles.reguig/These/Dataset/cat12/CATI/l0vs1234_y.csv"],

        "cati1": [
            "/network/lustre/iss01/cenir/analyse/irm/users/ghiles.reguig/These/Dataset/cat12/CATI/x_cati.csv",
            "/network/lustre/iss01/cenir/analyse/irm/users/ghiles.reguig/These/Dataset/cat12/CATI/l01vs234_y.csv"],

        "cati2": [
            "/network/lustre/iss01/cenir/analyse/irm/users/ghiles.reguig/These/Dataset/cat12/CATI/x_cati.csv",
            "/network/lustre/iss01/cenir/analyse/irm/users/ghiles.reguig/These/Dataset/cat12/CATI/l012vs34_y.csv"],

        "cati3": [
            "/network/lustre/iss01/cenir/analyse/irm/users/ghiles.reguig/These/Dataset/cat12/CATI/x_cati.csv",
            "/network/lustre/iss01/cenir/analyse/irm/users/ghiles.reguig/These/Dataset/cat12/CATI/l0123vs4_y.csv"],
    }

    datasets = {ds_name: read_dataset(ds_paths, drop_site=drop_site) for ds_name, ds_paths in paths.items()}

    return datasets

########################################################################################################################


def sensitivity(y_true, y_pred):
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    return sensitivity


def specificity(y_true, y_pred):
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    return specificity


def encode_vals(x):
    return LabelEncoder().fit_transform(x)


def roc_auc_threshold(y_true, y_pred):
    fpr, tpr, threshold = roc_curve(y_true, y_pred)
    imin = np.argmax(tpr - fpr)
    best_thr = threshold[imin]  # threshold[np.argmax(tpr/fpr)]
    return best_thr


def plot_f_test(all_ds, ft_names="MRIQC", save_path=None):
    margin = 0
    size = len(all_ds)
    plt.figure(figsize=(20, 15))
    for ds_name, data in all_ds.items():
        x, y = data["x"], data["y"]

        if "site" in x.columns:
            x = data["x"].copy()
            x["site"] = encode_vals(x["site"])
        x_data, y_data = x.values, y.values
        fval, pval = f_classif(x_data, y_data)
        ticks = np.array(range(0, size*len(fval)+len(fval), size+1))
        #### F VALUE
        #plt.figure(figsize=(20, 15))
        plt.barh(ticks + margin, fval, height=.50, label=ds_name)
        plt.yticks(ticks + margin, x.columns, rotation="horizontal")
        plt.ylabel("Feature")
        plt.xlabel("F-values")
        margin += 1.0
    plt.title("F-value of the {} features for various datasets".format(ft_names))
    plt.legend()
    if save_path:
        plt.savefig(opj(save_path, "fvals_{}_{}.png".format(ds_name, ft_names)))

    plt.figure(figsize=(20, 15))
    margin = 0
    for ds_name, data in all_ds.items():
        x, y = data["x"], data["y"]

        if "site" in x.columns:
            x = data["x"].copy()
            x["site"] = encode_vals(x["site"])
        x_data, y_data = x.values, y.values
        fval, pval = f_classif(x_data, y_data)
        ticks = np.array(range(0, size*len(pval)+len(pval), size+1))
        #### F VALUE
        #plt.figure(figsize=(20, 15))
        plt.barh(ticks + margin, 1-pval, height=.50, label=ds_name)
        plt.yticks(ticks + margin, x.columns, rotation="horizontal")
        plt.ylabel("Feature")
        plt.xlabel("1-P-value")
        margin += 1.0

    plt.title("P-values of the {} features for various datasets".format(ft_names))
    plt.legend()
    if save_path:
        plt.savefig(opj(save_path, "pvals_{}_{}.png".format(ds_name, ft_names)))

########################################################################################################################


class Pipeline_coef(Pipeline):

    def fit(self, X, y=None, **fit_params):
        super(Pipeline_coef, self).fit(X, y, **fit_params)
        model = self._final_estimator
        #print("Model : {}".format(model))
        
        if hasattr(model, "coef_"):
            self.coef_ = model.coef_

        if hasattr(model, "feature_importances_"):
            self.feature_importances_ = model.feature_importances_

    def _iter(self, with_final=True, filter_passthrough=True):
        """
        Generate (idx, (name, trans)) tuples from self.steps
        When filter_passthrough is True, 'passthrough' and None transformers
        are filtered out.
        """
        stop = len(self.steps)
        if not with_final:
            stop -= 1

        for idx, (name, trans) in enumerate(islice(self.steps, 0, stop)):
            if not filter_passthrough:
                yield idx, name, trans
            elif trans is not None and trans != 'passthrough':
                yield idx, name, trans

    def _transform(self, X):
        Xt = X
        for _, name, transform in self._iter(with_final=False):
            Xt = transform.transform(Xt)
        return Xt

########################################################################################################################


class ThresholdedClf(BaseEstimator, ClassifierMixin):

    def __init__(self, model=None, n_classes=2):
        self.model = model
        self.threshold = None
        self.n_classes = 2
        self._set_model_attrs()

    def _set_model_attrs(self):

        for att, val in self.model.__dict__.items():
            self.__setattr__(att, copy(val))

        if hasattr(self.model, "named_steps"):
            self.named_steps = self.model.named_steps

    def fit(self, X, y):
        self.model.fit(X, y)
        self._set_model_attrs()
        probs = self.model.predict_proba(X)

        if self.n_classes == 2 and len(probs.shape) == 2:
            probs = probs[:, 1]
        self.threshold = roc_auc_threshold(y, probs)
        return self

    def transform(self, X):
        return self.model.transform(X)

    def _transform(self, X):
        return self.model._transform(X)

    def predict(self, X):
        probs = self.model.predict_proba(X)
        if self.n_classes == 2 and len(probs.shape) == 2:
            probs = probs[:, 1]

        preds = probs > self.threshold
        return preds.astype(int)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

########################################################################################################################

