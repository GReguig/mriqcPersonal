import logging
import pickle
import pandas as pd
from os.path import isfile
from os.path import join as opj
import sys
from .crossvals import CVHelper
from .utils import read_dataset, read_all_datasets_mriqc, read_all_datasets_cat12
from .postprocessing import grid_save
from sklearn.model_selection import RepeatedStratifiedKFold


def get_parser():
    from argparse import ArgumentParser
    from argparse import RawTextHelpFormatter

    parser = ArgumentParser(description="Feature classification",
                            formatter_class=RawTextHelpFormatter)

    parser.add_argument("--train", nargs="*",
                        help="training data tables, X and Y")

    parser.add_argument("--test", nargs="*",
                        help="test data tables, X and Y")

    parser.add_argument("--model", action='store', default="svm",
                        choices=["svm", "rfc"])

    parser.add_argument("--folds", default=10,
                        help="Number of folds of cross-validation")

    parser.add_argument("--log-dir", default="",
                        help="Filepath to the logfile")

    parser.add_argument("--cv-file",
                        help="Filepath to the crossvalidation")

    parser.add_argument("--eval", choices=["cat12", "mriqc"],
                        help="Datasets for evaluation")
    
    parser.add_argument("--load", help="Filepath to the model to load")
    return parser


def __check_files(args):

    if args is None:
        raise RuntimeError("Please specify training data for the model")

    if len(args) != 2:
        raise RuntimeError("The --train needs 2 arguments")
    return args


def main():
    from . import logging, LOG_FORMAT

    drop_site = False
    #Parse arguments
    opts = get_parser().parse_args()
    #Set logger
    log = logging.getLogger("mriqcPersonal")
    dir_path = opts.log_dir
    #FileHandler
    fhl = logging.FileHandler(opj(dir_path, "cv_mriqc.log"))
    fhl.setFormatter(fmt=logging.Formatter(LOG_FORMAT))
    #fhl.setLevel(level=)
    log.addHandler(fhl)
    log.info("Starting program")
    #Model
    type_model = opts.model
    log.info("Found model : {}".format(type_model))
    __check_files(opts.train)
    dataset = read_dataset(opts.train, drop_site=drop_site)

    eval = opts.eval
    load = opts.load
    to_evaluate = None
    if eval == "cat12":
        to_evaluate = read_all_datasets_cat12(drop_site=drop_site)
        log.info("Using cat12 features, data will be evaluated on datasets: {} ".format(list(to_evaluate.keys())))
    elif eval == 'mriqc':
        to_evaluate = read_all_datasets_mriqc(drop_site=drop_site)
        log.info("Using mriqc features, data will be evaluated on datasets: {} ".format(list(to_evaluate.keys())))

    cv_file = opts.cv_file
    if load:
        with open(load, "rb") as f:
            cvhelper = pickle.load(f)
        print("Loaded : {}".format(cvhelper))
        splits = RepeatedStratifiedKFold(n_splits=10, n_repeats=1)
        grid_save(cvhelper, dataset["x"], dataset["y"], cv=splits, dir=dir_path, to_evaluate=to_evaluate, save_model=False)
    else:    
        cvhelper = CVHelper(type_model, dataset["x"], dataset["y"], cv_file=cv_file, log_dir=dir_path,
                            to_evaluate=to_evaluate)
    
        log.info("Fitting...")
        
        cvhelper = cvhelper.fit()
        cvhelper.evaluate()

if __name__ == '__main__':

    main()
