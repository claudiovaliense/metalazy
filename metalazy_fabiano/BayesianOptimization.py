import numpy as np
from hyperopt import fmin, tpe, STATUS_OK, Trials, space_eval
from sklearn.base import clone
from sklearn.model_selection import cross_val_score


class BayesianOptimization:
    def __init__(self, classifier, X, y, n_iter=20, scoring="f1_micro", cv=3, n_jobs=-1, verbose=0, random_state=42):
        self.classifier = classifier
        self.X = X
        self.y = y
        self.n_iter = n_iter
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = np.random.RandomState(random_state)

    def __repr__(self):
        return f"BayesianOptimization(classifier={self.classifier}, X={self.X.shape}, y={self.y.shape}, " \
               f"n_iter={self.n_iter}, scoring={self.scoring}, cv={self.cv}, n_jobs={self.n_jobs}, " \
               f"verbose={self.verbose}, random_state={self.random_state})"

    def __str__(self):
        return f"BayesianOptimization(classifier={self.classifier}, X={self.X.shape}, y={self.y.shape}, " \
               f"n_iter={self.n_iter}, scoring={self.scoring}, cv={self.cv}, n_jobs={self.n_jobs}, " \
               f"verbose={self.verbose}, random_state={self.random_state})"

    def __hyperopt_train_test(self, hyperparameters):
        classifier = clone(self.classifier)
        classifier.set_params(**hyperparameters)

        cv_metrics = cross_val_score(estimator=classifier, X=self.X, y=self.y, scoring=self.scoring, cv=self.cv,
                                     n_jobs=self.n_jobs, verbose=self.verbose)
        avg_metric = cv_metrics.mean()

        return avg_metric

    def __f(self, hyperparameters):
        avg_metric = self.__hyperopt_train_test(hyperparameters)
        loss_info = {"loss": -avg_metric, "status": STATUS_OK}

        return loss_info

    def fit(self, hyperparameters):
        best_parameters = fmin(fn=self.__f, space=hyperparameters, algo=tpe.suggest, max_evals=self.n_iter, rstate=self.random_state)

        best_parameters = space_eval(hyperparameters, best_parameters)

        return best_parameters
