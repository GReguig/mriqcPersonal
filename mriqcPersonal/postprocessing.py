import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from os.path import join as opj
from .utils import ThresholdedClf, specificity, sensitivity, roc_auc_threshold
from seaborn import kdeplot
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import permutation_test_score
from sklearn.metrics import balanced_accuracy_score, precision_score, accuracy_score, classification_report, roc_auc_score
from . import logging

LOG = logging.getLogger("mriqcPersonal")
LOG.setLevel(logging.INFO)


def evaluate_model(estimator, eval_x, eval_y, cv):
    n_permutations = 1#00
    sfm = SelectFromModel(estimator=estimator, prefit=True, max_features=10, threshold=-np.inf)
    sfm.transform(estimator._transform(eval_x))

    best_features = np.asarray(estimator.named_steps["adaptor"].columns)[sfm.get_support()]

    true_score, perm_scores, pval = permutation_test_score(estimator, eval_x, eval_y, scoring="roc_auc",
                                                           cv=cv, n_permutations=n_permutations, n_jobs=-1)
    LOG.info("Permutation test scores:\nFor {} permutations, p-value : {}\n".format(n_permutations, pval))
    LOG.info("Best features : {}".format(best_features))
    res = {
        "best_features": np.array2string(best_features),
        "ROC_AUC_score": true_score,
        "pval": pval,
        "perm_scores": np.array2string(perm_scores)
            }
    if hasattr(estimator, "threshold"):
        res["threshold"] = estimator.threshold
    return res


def compute_all_scores(model, datasets, scorers=[], log_dir=""):
    all_scores = {}
    """
    if hasattr(model, "threshold"):
        all_scores["threshold"] = model.threshold
    """
    alpha = 0.3
    for ds, data in datasets.items():
        x, y = data["x"], data["y"]
        probs = model.predict_proba(x)[:, 1]
        thresh = roc_auc_threshold(y.values, probs)
        preds = probs > thresh
        preds = preds.astype(int)
        #max_prob = np.max(probs)
        #alpha = .6 if max_prob < .5 else .3
        plt.hist(probs, bins=50, label=ds, alpha=alpha, density=True)

        #alpha += 0.03
        #preds = model.predict(x)

        scores_ds = {k: s(y, preds) for k, s in scorers.items()}
        scores_ds["threshold"] = thresh
        all_scores[ds] = scores_ds
        to_save = y.copy()
        to_save["Prob"] = probs
        to_save["Prediction"] = preds
        to_save.to_csv(opj(log_dir, "prediction_{}.csv".format(ds)))
        LOG.info("Classification report for {}\n{}\n\n".format(ds,
                                                               classification_report(y.values, preds,
                                                                                         target_names=["Non artifacted",
                                                                                                       "artifacted"])))
    plt.xlabel("Probability of being artifacted")
    plt.ylabel("Frequence")
    plt.title("Probability distribution of the MRIQC model for different datasets")
    plt.legend()

    plt.savefig(opj(log_dir, "datasets_probs.png"))
    plt.close("all")
    return all_scores


def grid_save(cvhelper, eval_x, eval_y, cv,  dir, sorting="rank_test_roc_auc", n_classes=2, to_evaluate=None, save_model=False):
    grid = cvhelper.grid
    best_estimator = ThresholdedClf(grid.best_estimator_, n_classes=n_classes).fit(eval_x, eval_y)
    ###### SAVE
    if save_model:
        with open(opj(dir, "model.p"), "wb") as f:
            pickle.dump(cvhelper, f)
         
    #ft_sc = best_estimator.named_steps["reduce"].scores_
    """
    df_scores = pd.DataFrame({"Feature": np.array(eval_x.columns), "Score": np.squeeze(ft_sc)})
    df_scores.to_csv(opj(dir, "ft_scores.csv"), index=False)
    """
    """
    plt.figure(figsize=(20, 15))
    plt.bar(range(len(ft_sc)), ft_sc)
    plt.xticks(range(len(ft_sc)), eval_x.columns, rotation="vertical")
    plt.savefig(opj(dir, "test_ft_scores.png", ))
    plt.close("all")
    """
    if to_evaluate:
        scorers = {"weighted_accuracy": balanced_accuracy_score,
                   "accuracy": accuracy_score,
                   "precision": precision_score,
                   "sensitivity": sensitivity,
                   "specificity": specificity,
                   "roc_auc": roc_auc_score,
                   }

        all_scores = compute_all_scores(best_estimator, to_evaluate, scorers=scorers, log_dir=dir)
        df_all_scores = pd.DataFrame.from_dict(all_scores)
        df_all_scores.to_csv(opj(dir, "evaluation_res_thr_{:.3f}.csv".format(best_estimator.threshold)))
        LOG.info("all_scores: \n{}".format(pd.DataFrame.from_dict(all_scores)))

    cv_results = grid.cv_results_
    df_cv = pd.DataFrame.from_dict(cv_results).sort_values(by=sorting)
    df_cv.to_csv(opj(dir, "cv_results.csv"), index=False)
    save_best_scores(df_cv, dir)
    """
    eval_dict = evaluate_model(best_estimator, eval_x, eval_y, cv)
    save_best_scores(df_cv, dir, eval_dict["best_features"])
    plot_perm_hist(eval_dict["ROC_AUC_score"], eval_dict["perm_scores"], 
                   eval_dict["pval"], n_classes=n_classes)
    plt.savefig(opj(dir, "permutation_scores.png"))
    
    plt.close("all")
    """


def save_best_scores(cv_results, dir):
    cols_to_keep = [c for c in cv_results.columns if c.find("mean") != -1
                                                  or c.find("std") != -1]
    cols_to_keep.append("params")
    best_estimator_res = cv_results.iloc[0][cols_to_keep]
    best_estimator_res.to_csv(opj(dir, "best_estimator_res.csv"))
    

def plot_perm_hist(score, permutation_scores, pvalue, n_classes=2):
    plt.hist(permutation_scores, bins=50, label='Permutation scores',
             edgecolor='black')
    ylim = plt.ylim()
    plt.plot(2 * [score], ylim, '--g', linewidth=3,
             label='Classification Score'
                   ' (pvalue %s)' % pvalue)
    plt.plot(2 * [1. / n_classes], ylim, '--k', linewidth=3, label='Luck')
    plt.title("Permutation scores of the best model")
    plt.xlabel('Score')
    plt.ylim(ylim)
    plt.yticks(np.linspace(0.0, 1.0, num=11))
    plt.legend()
    

"""
self.best_estimator = grid.best_estimator_#.named_steps[self.model_name]
df_res = pd.DataFrame.from_dict(self.cv_results).sort_values(by="rank_test_roc_auc")

LOG.info("CV Results:\n{}".format(self.cv_results))
LOG.info("Saving CV results in {}".format(opj(self.log_dir, "cv_results.csv")))
df_res.to_csv(opj(self.log_dir, "cv_results.csv"), index=False)

#Feature selection: Show the 10 best features
LOG.info("Determining best features of model...")
sfm = SelectFromModel(self.best_estimator, prefit=True, max_features=10, threshold=-np.inf)
sfm.transform(self.x_df)

selected_features = self.ftnames.where(sfm.get_support()).dropna().values
LOG.info("Computing permutation test score for the best model...")
permutation_score = permutation_test_score(self.best_estimator, self.x_df, self.ytrain, scoring="roc_auc",
                                           cv=splits, n_permutations=10, n_jobs=-1)
"""