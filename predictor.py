import pandas as pd
import numpy as np

from data import Dataprocess
import xgboost as xgb

class Model(object):

    def makesubmission(self, predict_y, savename="submit.csv"):
        submit_df = pd.read_csv("sample_submission.csv")
        submit_df["shot_made_flag"] = predict_y
        #submit_df = submit_df.fillna(np.nanmean(predict_y))
        submit_df.to_csv(savename, index=False)

#	def create_feature_map(features):
#		outfile = open('xgb.fmap', 'w')
#		for i, feat in enumerate(features):
#			outfile.write('{0}\t{1}\tq\n'.format(i, feat))
#		outfile.close()


    def xgboost(self, train_X, train_y, test_X, params=None, num_boost_round= 30):
        if params is None:
		params = {'objective': 'multi:softprob',
				  'eval_metric': 'mlogloss',
				  'colsample_bytree': 0.3,
				  'min_child_weight': 3.5, 
				  'subsample': 1.0, 
				  'eta': 0.2, 
				  'max_depth': 7, 
				  'gamma':3.1,
				  'num_class': 2,
				  'n_estimators': 520.0
		}
        dtrain = xgb.DMatrix(train_X, label=train_y)
        dtest = xgb.DMatrix(test_X)
        self.bst = xgb.train(params, dtrain, num_boost_round=num_boost_round)	
        test_y = self.bst.predict(dtest)
        return test_y


if __name__ == "__main__":
    dp = Dataprocess()
    train_X, train_y, test_X = dp.process()
    Model = Model()
    y = Model.xgboost(train_X, train_y, test_X)
    Model.makesubmission(y[0:,1], savename="submitAdhoc.csv")


