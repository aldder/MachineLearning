from hyperopt import tpe, space_eval, Trials
from numpy.random import RandomState
import matplotlib.pyplot as plt
from hyperopt.fmin import fmin
from time import time
import pandas as pd
import numpy as np
import sklearn
import joblib
import os


class HyperoptSearchCV(sklearn.base.BaseEstimator):

    def __init__(self, estimator, search_space, param_types={},
                 n_iter=25, scoring='accuracy', cv=None,
                 verbose=1, seed=42, n_jobs=1, greater_is_better=False, 
                 parser=None, trials_path=None):
        """ Constructor for model to be optimized using hyperopt

        Keyword arguments:
            :param object estimator: -- model with sklearns conventional interface (fit(), predict())
            :param dict search_space: -- dictionary with search space of parameters for hyperopt
            :param dict param_types:-- dictionary with types to cast, `None` - for no casting
            :param int n_iter: -- integer max number of evaluations
            :param object scoring:-- string or function
            :param int cv: -- int, cross-validation generator or an iterable, optional
            :param bool verbose: -- boolean, True for printing log
            :param int seed: -- seed for hyperopt `tpe` optimization function
            :param int n_jobs: -- int, number of cpu to use (-1 to use all cpus)
            :param bool greater_is_better: -- boolean, according to the score function
            :param function parser: -- function to parse the search space before optimization
            :param str trials_path: -- path to save or load an existing trial file, can be an existing trial file to restore a previously interrupted optimization
        """
        self.estimator = estimator
        self.search_space = search_space
        self.param_types = param_types
        self.scoring = scoring
        self.cv = cv
        self.verbose = verbose
        self.n_iter = n_iter
        self.seed = seed
        self.best_params_ = None
        self.best_score_ = None
        self.X_train = None
        self.y_train = None
        self.n_jobs = n_jobs
        self.greater_is_better = greater_is_better
        self.parser = parser
        self.trials_path = trials_path


    def __cast_params(self, recv_params):
        # cast the parameters stored in `recv_params` to
        # types stored in `self.param_types`
        casted_params = {}
        for param_name, param_value in recv_params.items():
            param_type = self.param_types.get(param_name, None)  # if type for casting not found, skip
            if param_type is None:
                casted_params[param_name] = param_value
            else:
                casted_params[param_name] = param_type(param_value)

        return casted_params


    def __objective(self, recv_params):
        casted_params = self.__cast_params(recv_params)
        
        if self.parser:
            casted_params = self.parser(casted_params)
        
        updated_model = self.estimator.set_params(**casted_params)
        
        tic = time()
        
        score = sklearn.model_selection.cross_val_score(
            updated_model,
            self.X_train,
            self.y_train,
            scoring=self.scoring,
            cv=self.cv,
            n_jobs=self.n_jobs).mean()
        
        toc = time()

        if self.greater_is_better:
            score = -score
        
        if self.verbose:
            print(f"{score:.3f} ({toc-tic:.2f}s) - mean score on CV with params {recv_params}")

        if score < self.best_score_:
            self.best_score_ = score
        return score


    def fit(self, X_train, y_train):

        if self.best_score_ is None:
            self.best_score_ = float('inf')

        if self.trials_path is not None and os.path.exists(self.trials_path):
            self.trials_obj = joblib.load(self.trials_path)
        else:
            self.trials_obj = Trials()
        
        self.X_train = X_train
        self.y_train = y_train

        self.estimator = sklearn.base.clone(self.estimator)

        best_params = fmin(fn=self.__objective,
                           space=self.search_space,
                           algo=tpe.suggest,
                           max_evals=self.n_iter,
                           trials=self.trials_obj,
                           rstate=RandomState(42))
        evaluated = space_eval(self.search_space, best_params)
        
        print('best params: ', evaluated)
        
        joblib.dump(self.trials_obj, self.trials_path)
        
        casted_best_params = self.__cast_params(evaluated)
        if self.parser:
            casted_best_params = self.parser(casted_best_params)
        self.best_params_ = casted_best_params

        self.estimator.set_params(**casted_best_params)
        self.estimator.fit(self.X_train, self.y_train)
        
        return self


    def predict(self, X):
        return self.estimator.predict(X)


    def predict_proba(self, X):
        return self.estimator.predict_proba(X)


    def visualize_trials(self, max_loss=None):
        
        trials_df = pd.DataFrame(columns=self.trials_obj.trials[0]['misc']['vals'].keys())
        
        if max_loss:
            valid_trials = [trial for trial in self.trials_obj.trials if trial['result']['loss'] <= max_loss]
        else:
            valid_trials = self.trials_obj.trials
        
        for i, trial in enumerate(valid_trials):
            trials_df.loc[i, 'error'] = trial['result']['loss']
        
        for i, trial in enumerate(valid_trials):
            for k in trial['misc']['vals'].keys():
                trials_df.loc[i, k] = trial['misc']['vals'][k][0] if len(trial['misc']['vals'][k]) > 0 else np.nan

        for i,c in enumerate(trials_df.drop(['error'],1).columns):
            fig, ax = plt.subplots(1,2, figsize=(8, 3))
            ax[0].scatter(trials_df.index, trials_df[c], s=20, linewidth=0.01, alpha=0.75)
            ax[0].set_title(f'{c} over trials')
            ax[0].set_xlabel('trial')
            ax[0].set_ylabel(c)
            ax[1].scatter(trials_df[c], trials_df['error'], s=20, linewidth=0.01, alpha=0.75)
            ax[1].set_title(f'error over {c}')
            ax[1].set_xlabel(c)
            ax[1].set_ylabel('error')
            plt.tight_layout()
            # plt.show()

        trial_seq = []
        err_seq = []
        for trial in valid_trials:
            trial_seq.append(trial['tid'])
            err_seq.append(trial['result']['loss'])
        plt.figure(figsize=(9,3))
        plt.plot(trial_seq, err_seq)
        plt.scatter(np.argmin(err_seq), min(err_seq), c='r')
        plt.title('error over trials')
        plt.xlabel('trial')
        plt.ylabel('error')
        plt.show()
