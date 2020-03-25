from .custom_gridsearch import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold, permutation_test_score
from .utils import Pipeline_coef as Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectPercentile
import numpy as np
from . import logging
from .preprocessing import ColumnsScaler, PandasAdaptor, BatchRobustScaler, SiteCorrelationSelector, CustFsNoiseWinnow
from .postprocessing import grid_save
from os.path import join as opj

LOG = logging.getLogger("mriqcPersonal")
LOG.setLevel(logging.INFO)


class CVHelper(object):

    def __init__(self, model_name, x_df, y_df, n_jobs=-1, to_evaluate=None, cv_file=None, log_dir=""):
        self.x_df = x_df
        self.y_df = y_df
        self.ftnames = x_df.columns
        self.xtrain = x_df.values
        self.ytrain = y_df["y"].values
        self.model_name = str.lower(model_name)
        self.model = self._read_model()
        self.n_jobs = n_jobs
        self.to_evaluate = to_evaluate
        self.cv_results = None
        self.best_estimator = None
        self.cv_file = cv_file
        self.log_dir = log_dir
        self.n_classes = len(np.unique(self.ytrain))
        self.n_features = x_df.shape[1]

    def fit(self):
        #Cross Validation Split
        self.splits = RepeatedStratifiedKFold(n_splits=10, n_repeats=3)
        #Parameters of the CV
        params = self.get_parameters()
        LOG.info("GridSearch parameters\n{}".format(params))
        pipeline = Pipeline(
            [
                #("scaler", ColumnsScaler()),
                ("scaler", BatchRobustScaler()),
                #("ft_select", RFE())
                ("adaptor", PandasAdaptor()),
                ("ft_site", SiteCorrelationSelector(max_iter=200)),
                ("ft_noise", CustFsNoiseWinnow()),
                (self.model_name, self.model)
            ])
        LOG.info("Pipeline : {}".format(pipeline.named_steps))
        #Do the gridsearch cross Validation
        grid = GridSearchCV(pipeline, params, error_score=0.5, refit="roc_auc", cv=self.splits, n_jobs=self.n_jobs,
                            scoring=["roc_auc", "accuracy", "precision", "recall", "f1"], to_evaluate=self.to_evaluate)
        #grid.fit(self.xtrain, self.ytrain)
        grid.fit(self.x_df, self.ytrain)
        
        self.grid = grid
        self.best_estimator = grid.best_estimator_
        return self

    def evaluate(self):
        grid_save(self, self.x_df, self.ytrain, cv=self.splits, dir=self.log_dir, 
                  sorting="rank_test_roc_auc", n_classes=self.n_classes,
                  to_evaluate=self.to_evaluate, save_model=True)
        
        LOG.info("Best model\n{}".format(self.best_estimator))

    def get_parameters(self):
        params = self._read_params()
        output = dict()
        for key, val in params.items():
            output.update(val[0])
        #output["scaler__scaler"] = [StandardScaler()]
        #output["scaler__columns"]=[["cjv", "cnr", "efc"]]
        return output

    def _read_params(self):
        import yaml
        with open(self.cv_file) as paramfile:
            parameters = yaml.load(paramfile)

        if self.model_name == "rfc":
            del parameters["svm"]

        elif self.model_name == "svm":
            del parameters["rfc"]

        return parameters

    def _read_model(self):
        if self.model_name == "rfc":
            return RandomForestClassifier()

        elif self.model_name == "svm":
            return SVC()
