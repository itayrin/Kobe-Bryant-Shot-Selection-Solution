
from data import Dataprocess
from tuning import Tuner
from predictor import Model
import numpy as np
import xgboost as xgb
import json

# Ref : https://github.com/dmlc/xgboost/blob/master/demo/multiclass_classification/train.py#L2

#	This section consists of two phases. The first phase is exploration based on 
#	native data base provided. The second phase is of imputing prior prediction as_matrix
#	and then running the classifier. The second phase is the stacking method. A single
#   layer of stacking is implemneted in this code,  	

if __name__ == "__main__":
	dp = Dataprocess()
	train_X, train_y, test_X = dp.process()
	Tuner = Tuner()
	params, results = Tuner.tune(train_X, train_y, test_X, max_evals=2500)
	# Export the best param fitted as json format
	json.dump(param,open("param.txt",'w'))
	np.random.seed(0)
	Model = Model()

	test_y = Model.xgboost(train_X, train_y, test_X, params=params, num_boost_round=30)
	# This is the first phase of submission : This is prior the stacking method
	# applied to the model.
	Model.makesubmission(test_y[0:,1], savename="fist_submit.csv")


	# Stacking method applied : Single Layer Stacking 
	all_X = dp.mapper_X.transform(dp.df)
	all_y = Model.xgboost(train_X, train_y, all_X, num_boost_round=30)
	plus_X = np.append(all_X, all_y, axis=1)
	train_X_plus = plus_X[~np.isnan(dp.df["shot_made_flag"].as_matrix()), :]
	test_X_plus = plus_X[np.isnan(dp.df["shot_made_flag"].as_matrix()), :]

	Tuner_plus = tuning.Tuner()
	params_plus, results_plus = Tuner_plus.tune(train_X_plus, train_y, test_X_plus, max_evals=2500)
	# Export the best param fitted as json format
	json.dump(param,open("finParam.txt",'w'))

	test_y_plus = Model.xgboost(train_X_plus, train_y, test_X_plus, params=params_plus, num_boost_round=30)
	Model.makesubmission(test_y_plus[0:,1], savename="final_submit.csv")






