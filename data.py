import pandas as pd
import numpy as np

from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelBinarizer
from sklearn import mixture


class Dataprocess(object):
    def read(self):
        self.df_orig = pd.read_csv("data.csv")
        self.df = self.df_orig.copy()

    def process(self):
        self.read()
        self.preproc()
        self.set_mapper()
        self.split_df()
        train_X = self.vec_X(self.train_df)
        train_y = self.vec_y(self.train_df)
        test_X = self.mapper_X.transform(self.test_df)
        return train_X, train_y, test_X

    def preproc(self):													# Considering NBA Rules: 
	   	self.df['time_remaining'] = self.df['minutes_remaining'] * 60 + self.df['seconds_remaining']		# compute the total remaining time from scoring/attempt in a 12 minute (720seconds) session
		self.df['last_3_sec'] = self.df['time_remaining'] < 3							# The atheletes up their ante at the last seconds and hence an important fetaure 
		self.df['last_5_sec'] = self.df['time_remaining'] < 5							# -do- 	 Longer shots and mostly missed in this zone	
		self.df['earlier_half'] = self.df['time_remaining'] < 90
		self.df['later_half'] = self.df['time_remaining'] < 180
		self.df['last_quarter'] = self.df['time_remaining'] < 360
		self.df['secondsFromStart'] = 720 - self.df["time_remaining"]

		# This encompasses all the periods, based on data exploration diagrams	
		# Based on data exploration, the fields are distinct till 4th periods
		# Then upon 5 period and henceforth are left in final_period	
		self.df['first_period'] = self.df['period'] == 1
		self.df['second_period'] = self.df['period'] == 2
		self.df['third_period'] = self.df['period'] == 3
		self.df['fourth_period'] = self.df['period'] == 4
		self.df['final_period'] = self.df['period'] > 4	

		# Based on data exploration, we may understand that the EM algorithm need not determine 
		# marginal distributions beyond the required necessity. Hence 8 distribution subset is allowed. 
		# This may however be tweaked as a knob during optimization : (5,13)
		gaussianMixtureModel1 = mixture.GMM(n_components= 13, covariance_type='full', 
					               params='wmc', init_params='wmc',
					               random_state=1, n_init=3,  verbose=0)
		gaussianMixtureModel1.fit(self.df.ix[:,['loc_x','loc_y']])
		self.df['shotLocationCluster1'] = gaussianMixtureModel1.predict(self.df.ix[:,['loc_x','loc_y']])		

		# suspect this to be correlated to loc_x and loc_y	
		gaussianMixtureModel2 = mixture.GMM(n_components= 13, covariance_type='full', 
					               params='wmc', init_params='wmc',
					               random_state=1, n_init=3,  verbose=0)
		gaussianMixtureModel2.fit(self.df.ix[:,['lat','lon']])
		self.df['shotLocationCluster2'] = gaussianMixtureModel2.predict(self.df.ix[:,['lat','lon']])						

		# Dates are split into month, year and days for more vector fields
		# These variables should reflect in reduced feature extraction from Random Forests 
		# https://en.wikipedia.org/wiki/List_of_career_achievements_by_Kobe_Bryant
		# 2000-2002 (career high) / playoff
		# 2005-2006 (highest point per game)
		# 2007-2010 (career high) /Playoff   
		self.df["game_year"] = pd.Series([int(self.df["game_date"][i][:4]) for i in range(0, len(self.df))])			
		self.df["game_month"] = pd.Series([int(self.df["game_date"][i][5:7]) for i in range(0, len(self.df))])
		self.df["game_day"] = pd.Series([int(self.df["game_date"][i][-2:]) for i in range(0, len(self.df))])

		# Range: 15 25  
		gameGaussianMixtureModel = mixture.GMM(n_components=25, covariance_type='full', 
					               params='wmc', init_params='wmc',
					               random_state=1, n_init=3,  verbose=0)
		gameGaussianMixtureModel.fit(self.df.ix[:,['game_id','game_event_id']])
		self.df['gameIdandEventCluster'] = gameGaussianMixtureModel.predict(self.df.ix[:,['game_id','game_event_id']])		


    def set_mapper(self):
		self.mapper_X = DataFrameMapper([
			(u'action_type', LabelBinarizer()),
			(u'combined_shot_type', LabelBinarizer()),

			(u'game_id', None),
			(u'game_event_id', None),
			(u'loc_x', None),
			(u'loc_y', None),
			(u'lat',None),
			(u'lon',None),
			(u'minutes_remaining', None),
			(u'period', LabelBinarizer()),

			(u'playoffs', LabelBinarizer()),
			(u'season', LabelBinarizer()),
			(u'seconds_remaining', None),
			(u'shot_distance', LabelBinarizer()),			
			(u'shot_type', LabelBinarizer()),
			(u'shot_zone_area', LabelBinarizer()),
			(u'shot_zone_basic', LabelBinarizer()),
			(u'shot_zone_range', LabelBinarizer()),
			(u'matchup', LabelBinarizer()),
			(u'shot_id', None),

			(u'season', LabelBinarizer()),
			(u'game_year', LabelBinarizer()),				
			(u'game_month', None),
			(u'game_day', None),

			(u'first_period', LabelBinarizer()),
			(u'second_period', LabelBinarizer()),
			(u'third_period', LabelBinarizer()),
			(u'fourth_period', LabelBinarizer()),
			(u'final_period', LabelBinarizer()),

			(u'time_remaining', None),
			(u'last_3_sec', LabelBinarizer()),
			(u'last_5_sec', LabelBinarizer()),
			(u'earlier_half', LabelBinarizer()),
			(u'later_half', LabelBinarizer()),
			(u'last_quarter', LabelBinarizer()),
			(u'secondsFromStart', None),
			(u'opponent', LabelBinarizer()),
			(u'game_id', LabelBinarizer()),

			(u'shotLocationCluster1', LabelBinarizer()),
			(u'shotLocationCluster1', LabelBinarizer()),
			(u'gameIdandEventCluster', LabelBinarizer()),
			])
		self.mapper_y = DataFrameMapper([(u'shot_made_flag', None),])
		
		self.mapper_X.fit(self.df)
		self.mapper_y.fit(self.df)

    def split_df(self):
        self.train_df = self.df[~np.isnan(self.df["shot_made_flag"])]
        self.test_df = self.df[np.isnan(self.df["shot_made_flag"])]


    def vec_X(self, df):
        return self.mapper_X.transform(df.copy())


    def vec_y(self, df):
        return self.mapper_y.transform(df.copy())


	# This function format is kept deliberate for Jupyter Notebook complaint format. The 
	# code set is left here for refernece
	def dataExplore(self):
		'''
		action_type            object
		combined_shot_type     object
		game_event_id           int64
		game_id                 int64


		period                  int64
		playoffs                int64
		season                 object
		shot_distance           int64

		shot_type              object
		shot_zone_area         object
		shot_zone_basic        object
		shot_zone_range        object

		game_date              object
		matchup                object
		opponent               object


		#f, axarr = plt.subplots(3, figsize=(25, 35))
		#sns.countplot(y="action_type", hue="shot_made_flag", data=data, palette="husl", ax=axarr[0]);
		#sns.countplot(y="game_event_id", hue="shot_made_flag", data=data, palette="cubehelix", ax=axarr[1]);
		#sns.countplot(y="game_id", hue="shot_made_flag", data=data, palette="Blues_d", ax=axarr[2]);


		#sns.countplot(y="combined_shot_type", hue="shot_made_flag", data=data, palette="Blues_d", ax=axarr[0]);
		#sns.countplot(y="period", hue="shot_made_flag", data=data, palette="husl", ax=axarr[1]);
		#sns.countplot(y="playoffs", hue="shot_made_flag", data=data, palette="Blues_d", ax=axarr[2]);
		#sns.countplot(y="season", hue="shot_made_flag", data=data, palette="coolwarm", ax=axarr[3]);



		#sns.countplot(y="shot_type", hue="shot_made_flag", data=data, palette="cubehelix", ax=axarr[0]);
		#sns.countplot(y="shot_zone_area", hue="shot_made_flag", data=data, palette="Blues_d", ax=axarr[1]);
		#sns.countplot(y="shot_zone_basic", hue="shot_made_flag", data=data, palette="cubehelix", ax=axarr[2]);
		#sns.countplot(y="shot_zone_range", hue="shot_made_flag", data=data, palette="Blues_d", ax=axarr[3]);
		#sns.countplot(y="opponent", hue="shot_made_flag", data=data, palette="Blues_d", ax=axarr[4]);


		#sns.countplot(y="shot_distance", hue="shot_made_flag", data=data, palette="Blues_d", ax=axarr[0]);
		#sns.countplot(y="game_date", hue="shot_made_flag", data=data, palette="BrBG", ax=axarr[1]);
		#sns.countplot(y="matchup", hue="shot_made_flag", data=data, palette="coolwarm", ax=axarr[2]);

		#axarr[0].set_title('action_type')
		#axarr[1].set_title('game_event_id')
		#axarr[2].set_title('game_id')

		#axarr[0].set_title('combined_shot_type')
		#axarr[1].set_title('period')
		#axarr[2].set_title('playoffs')
		#axarr[3].set_title('season')


		#axarr[0].set_title('shot_type')
		#axarr[1].set_title('shot_zone_area')
		#axarr[2].set_title('shot_zone_basic')
		#axarr[3].set_title('shot_zone_range')
		#axarr[4].set_title('opponent')

		#axarr[0].set_title('shot_distance')
		#axarr[1].set_title('game_date')
		#axarr[2].set_title('matchup')

		# Further exploration was done with due course in progress of the design
		#ndata= data[['timeRemain','shot_made_flag']]
		#sns.boxplot(x="shot_made_flag", y="timeRemain", data=ndata, hue ='shot_made_flag')	


		#listColData = list(data.columns.values)
		#var = data.shape
		#for i in range(0,(var[1])):
		#    print listColData[i],": ", len(data[listColData[i]].value_counts())


		#sns.pairplot(data, vars=['loc_x', 'loc_y', 'lat', 'lon', 'shot_distance','minutes_remaining','seconds_remaining'], 
															hue='shot_made_flag', palette=flatui, diag_kind="kde", size=3)
		#plt.show()	

		'''
		pass				# data exploration code is entirely commenetd out.




