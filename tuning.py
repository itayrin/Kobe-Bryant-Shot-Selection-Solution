import pandas as pd
import numpy as np


from sklearn import cross_validation
from sklearn.metrics import  log_loss
from sklearn.cross_validation import KFold
import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


class Tuner(object):
    def tune(self, train_X, train_y, test_X, max_evals=250):
        self.train_X = train_X
        self.train_y = train_y.reshape(len(train_y),)
        self.test_X = test_X
        np.random.seed(0)									# This method is called when RandomState is initialized. It can be called again to re-seed the generator
        trials = Trials()
        params = self.hyperParamOptimise(trials, max_evals=max_evals)
        params_result = self.score(params)
        return params, params_result


    def tuned_predict(self, params, num_boost_round, test_X=None):
        df_train_X = pd.DataFrame(self.train_X)
        if test_X is None:
            df_test_X = pd.DataFrame(self.test_X)
        else:
            df_test_X = pd.DataFrame(test_X)
        dtrain = xgb.DMatrix(df_train_X.as_matrix(), label=self.train_y.tolist())
        dtest = xgb.DMatrix(df_test_X.as_matrix())
        self.bst = xgb.train(params, dtrain, num_boost_round=num_boost_round)
        pred = self.bst.predict(dtest)
        return pred


    def score(self, params):
        print "Training with params : "
        print params
        N_boost_round=[]
        Score=[]
        skf = cross_validation.StratifiedKFold(self.train_y, n_folds=6, shuffle=True, random_state=25)
        for train, test in skf:
            X_Train, X_Test, y_Train, y_Test = self.train_X[train], self.train_X[test], self.train_y[train], self.train_y[test]

            dtrain = xgb.DMatrix(X_Train, label=y_Train)
            dvalid = xgb.DMatrix(X_Test, label=y_Test)
            watchlist = [(dtrain, 'train'),(dvalid, 'eval')]
            model = xgb.train(params, dtrain, num_boost_round=150, evals=watchlist, early_stopping_rounds=10)
            predictions = model.predict(dvalid)
            print "\n prediction.shape =", predictions.shape, "\n"
            N = model.best_iteration
            N_boost_round.append(N)
            score = model.best_score
            Score.append(score)
        Average_best_num_boost_round = np.average(N_boost_round)
        Average_best_score = np.average(Score)
        print "\tAverage of best iteration {0}\n".format(Average_best_num_boost_round)
        print "\tScore {0}\n\n".format(Average_best_score)
        return {'loss': Average_best_score, 'status': STATUS_OK, 'Average_best_num_boost_round': Average_best_num_boost_round}


    def hyperParamOptimise(self, trials, max_evals=250):
        self.space = {
	    "booster": 'gbtree',	
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            #Control complexity of model
            "eta" : hp.quniform("eta", 0.01, 0.06, 0.01),
            "max_depth" : 7,									
            "min_child_weight" : hp.quniform('min_child_weight', 2.0, 12.0, 0.5),
            'gamma' : hp.quniform('gamma', 0.1, 10.0, 0.1),
            #'learning_rate': hp.quniform('learning_rate', 0.3, 0.5, 0.1),
            'n_estimators': hp.quniform('n_estimators', 300, 800, 10),
            #Improve noise robustness 
            "subsample" : hp.quniform('subsample', 0.9, 1, 0.1),
            "colsample_bytree" : hp.quniform('colsample_bytree', 0.2, 0.6, 0.01),
            'num_class' : 2,
            'silent' : 1}
        best = fmin(self.score, self.space, algo=tpe.suggest, trials=trials, max_evals=max_evals)
	# We need to inject the non-optimizer default params as hyperopt seems to drop them from evaluation output
	best['objective'] = "multi:softprob"; best['booster'] = 'gbtree'; best['eval_metric'] = 'mlogloss'; best['num_class'] = 2; best['silent'] = 1	
        print "best parameters", best
        return best

