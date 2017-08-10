import pandas as pd
import numpy as np
import random
import warnings 
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import log_loss

#File csv
filename_train = 'puzzle_train_dataset.csv'
filename_test = 'puzzle_test_dataset.csv'
filename_result = 'predictions.csv'

names_models = ['Multinomial Naive Bayes','Naive Bayes', 'Random Forest Classifier','Nearest Neighbors' ,'AdaBoost','Linear SVM','Dummy']
type_models = [
	MultinomialNB(),
	GaussianNB(),
    RandomForestClassifier(max_depth=10, n_estimators=20, max_features=10),
	KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=1, n_neighbors=5, p=2, weights='uniform'),    
    AdaBoostClassifier(),
    SVC(kernel="linear", C=0.5, max_iter=15),
    DummyClassifier(strategy='most_frequent')
]

def read_file_csv(filename):
	df = pd.read_csv(filename, parse_dates=['last_payment','end_last_loan'])
	#Delete rows without value default
	df.dropna(axis=0, how='any', subset=['default'], inplace=True)
	return df

def read_file_test_csv(filename,feature_col):
	df = pd.read_csv(filename, parse_dates=['last_payment','end_last_loan'])
	data_test = handle_data_frame(df,feature_col)
	return data_test

#identify features, like categorical, numerical, boolean and date variables
def identification_variables(df):	
	ids_col = ['ids']
	target_col = ['default']
	categorical_col =['score_1','score_2','reason','state','job_name','real_state','zip','sign']
	date_col = ['last_payment','end_last_loan']
	boolean_col = ['gender', 'facebook_profile']
	repetitive_value_col = ['channel']
	numerical_col = list(set(list(df.columns)) - set(categorical_col)-set(ids_col)-set(target_col) -set(repetitive_value_col) -set(boolean_col)-set(date_col))
	date_col_add = ['delta_anticipate_payment','delta_late_payment']
	feature_col = numerical_col + categorical_col + boolean_col + date_col_add
	return numerical_col,categorical_col,feature_col


#Convert true and false values in binary values and remove rows that are above the threshold
def clean_handle_data_frame(df,income_threshold,numerical_col,categorical_col,feature_col):
	#first working the outlier, filter the rows with income < 124000
	df = df.query('income < %f' % income_threshold)

	#Convert true and false to value 0 and 1
	df['default'] = df['default'].apply(lambda x: x*1)

	data = handle_data_frame(df,feature_col)

	return data

#identify the variables with missing values and handle them, i.e transform string data in numeric data
#and fill values 
def handle_data_frame(df,feature_col):
	# Fill missing values with mean to numerical columns
	df[numerical_col] = df[numerical_col].fillna(df[numerical_col].mean(), inplace=True)
	# Fill missing values with value -99999
	df[categorical_col] = df[categorical_col].fillna(value = -99999)
	# convert string data in number data
	for var in categorical_col:
		number_label = LabelEncoder()
		df[var] = number_label.fit_transform(df[var].astype('str'))
	#convert boolean data in binary values
	df['facebook_profile'] = df['facebook_profile'].apply(lambda x: 0 if pd.isnull(x) else x*1)
	#Convert "m" and "f" to value 1 and 0
	df['gender'] = df['gender'].apply(lambda x: 0.5 if pd.isnull(x) else 1 if x == "m" else 0)

	# # Fill missing values of last_payment and end_last_loan with minimal values
	min_last_payment, min_end_last_loan  = df.last_payment.min(), df.end_last_loan.min()
	df = df.fillna({'last_payment':min_last_payment, 'end_last_loan':min_end_last_loan} )
    
    # Create new features related to end_last_loan and last_payment
	df['delta']  = ( df.end_last_loan  - df.last_payment).astype('timedelta64[D]')
	df['delta_anticipate_payment'] = df['delta'].where(df['delta'] >= 0, 0)
	df['delta_late_payment'] = df['delta'].where(df['delta'] < 0, 0).abs()
	
	return df

# split the size of train data set
def split_train_validate_set_data(df,feature_col):
	#Determine the set to train
	df['to_train'] = np.random.uniform(0, 1,len(df)) <= .8
	#Determine set to train and validate
	train_set, validate_set = df[ df['to_train'] == True ], df[ df['to_train'] == False]
	X_train =  train_set[list(feature_col)].values
	Y_train =  train_set['default'].values
	X_validate = validate_set[list(feature_col)].values
	Y_validate = validate_set['default'].values
	print("Total size:",len(X_train)+len(X_validate))
	print("Train size: " , len(X_train))
	print("Validate size: " , len(X_validate))
	return X_train, Y_train, X_validate, Y_validate

# Obtain the best model to be used in test dataset
def select_best_model(X_train, Y_train, X_validate, Y_validate):
	minLogLoss = 100
	best_model = None
	best_model_name = ""
	for name, model in zip(names_models, type_models):
		model.fit(X_train, Y_train)
		result_model = model.predict(X_validate)
		logLoss = log_loss(Y_validate, result_model)
		if logLoss < minLogLoss: 
			minLogLoss = logLoss
			best_model = model
			best_model_name = name
		print("Model name: " , name, ", logLoss: " , logLoss)
		
	print("\nModel with best result: ", best_model_name, ", Min LogLoss: ", minLogLoss)
	return best_model


def training_data(df,feature_col):
	X_train, Y_train, X_validate, Y_validate = split_train_validate_set_data(df,feature_col)
	model = select_best_model(X_train, Y_train, X_validate, Y_validate)
	return model

def make_prediction(model, feature_col):
	df_test = read_file_test_csv(filename_test,feature_col)
	X_test = df_test[feature_col].values
	result = model.predict_proba(X_test)
	return result, df_test

def write_result(result, df_test):
	df_test['predictions'] = result[:,0]
	df_result = df_test.loc[:,['ids','predictions']]
	df_result.to_csv(path_or_buf=filename_result, index=False)
	print("File created: ", filename_result)


if __name__ == "__main__":
	print("Start")
	df = read_file_csv(filename_train)
	income_threshold = 124000
	numerical_col, categorical_col,feature_col = identification_variables(df)
	df = clean_handle_data_frame(df,income_threshold,numerical_col,categorical_col,feature_col)

	model = training_data(df,feature_col)
	result, df_test = make_prediction(model,feature_col)
	write_result(result, df_test)
	print("End")
