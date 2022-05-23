import numpy as np 
import pandas as pd
import math as math
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

class AIC_Tab:
	def __init__(self, model_names, model_set, data):
		self.model_set = model_set
		self.model_estimates= [models.params for models in self.model_set] 
		self.df = data
		self.bic = [models.bic for models in model_set] 
		self.delta_bic = [x-min(self.bic) for x in self.bic]
		self.aic = [models.aic for models in model_set] 
		self.delta_aic = [x-min(self.aic) for x in self.aic]
		self.exp_delta_aic = [math.exp(-0.5 * x) for x in self.delta_aic]
		self.aic_weight = [x/np.nansum(self.exp_delta_aic) for x in self.exp_delta_aic]
		self.aic_cum_weight = np.nancumsum(self.aic_weight)
		self.ev_ratio = [max(self.aic_weight)/x for x in self.aic_weight]	
		self.log_likelihood= [models.llf for models in model_set] 
		self.aictab = pd.DataFrame(index= model_names)
		self.aictab['BIC'] = self.bic
		self.aictab['\u0394_BIC'] = self.delta_bic
		self.aictab['AIC'] = self.aic
		self.aictab['\u0394_AIC'] = self.delta_aic
		self.aictab['Log_Likelihood'] = self.log_likelihood
		self.aictab["exp(-0.5 * \u0394_AIC)"] = self.exp_delta_aic
		self.aictab['AIC_Weight'] = self.aic_weight
		self.aictab['AIC_Cum_Weight'] = self.aic_cum_weight
		self.aictab['Evidence_Ratios'] = self.ev_ratio
		# add and hide model estimates
		self.aictab = self.aictab.sort_values(by=['\u0394_AIC'], ascending = True)
		self.aictab = self.aictab.round(4)

	def instructions(self):
		print('FIRST: you must clean the data and make sure there are no missing data and that the samples the same and  balanced for each row. Print AIC_Tab.check() for a quick check', wrap = True)

	def check(self):
		self.describe = self.df.describe()
		self.counts = self.describe.loc['count', :]
		self.drop = self.counts.sort_values(ascending = True)
		self.balanced = (self.counts[0] == self.counts).all()
		self.to_drop = str(self.drop.index[0])
		if bool(self.balanced) == True:
			print('\n\n\n Well done! Your dataset is clean and your sample is balanced. To continue the model-selection process print the AIC.Tab.aictable(). You can use best_fit() to print the single best-fitmodel, if one is identified, or use  best_ranked() to use the 95% best ranked models.\n')
		else:
			print("\n\n\n Your sample is not balanced. Please consider dropping rows and columns with missing data. For example, {} has the lowest sample size. if this is an important parameter, keep all columns but drop the necessary rows with missing data. Otherwise drop the {} column completely.\n".format(self.to_drop, self.to_drop, wrap=True))

	def aictable(self):
		self.aictable = self.aictab.reset_index()
		self.aictable = self.aictable.rename(columns={'index': "Model_Name"})
		return self.aictable

	def best_fit(self):
		self.best = self.aictab[self.aictab['AIC_Weight'] >= 0.90]
		self.best_name = self.best.index[0]
		if bool(self.best_name) == None:
			print('\n\n\n Several models in your set best explained the data. You can print best_ranked() to identify the the 95% best ranked models and then model_averaging() to produce model-averaged parameter estimates.\n',wrap= True)
		else:
			print('\n\n\n {} was identified as being the single best-fit model. Print the model summary (model.summary()) and use the estimates to make inferences.\n'.format(self.best_name,wrap =True))

	def best_ranked(self):
		self.best = aictable()
		self.best_95 = self.aictable[self.aictable['AIC_Cum_Weight'] >= 0.95]
		return self.best_95
		# if nothing is returned state that inferences should be made on the single best fit model identified by using the best_fit() method.
	
	def model_averaging(self):
		self.model_estimates = [models.params for models in model_set] 
		self.model_sd = [models.bse for models in model_set]
		pass

	def crossval(self, X, y, classification):
		self.classification = classification
		self.X = self.df.loc[:, X]
		self.y = self.df.loc[:, y]

		# Preprocessing the data done by user
		self.enc = preprocessing.OrdinalEncoder()
		self.X = self.enc.fit_transform(self.X)
		self.y = self.enc.fit_transform(self.y)
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=0)

		if bool(self.classification) == True:
			# Logistic regression
			self.logistic_classifier = LogisticRegression(random_state = 0)
			self.logist=self.logistic_classifier.fit(self.X_train,self.y_train)
			self.logist_pred = self.logist.predict(self.X_test)
			self.logist_accuracy = accuracy_score(self.y_test, self.logist_pred)
		
			#Naive Bayes
			self.nb_classifier = MultinomialNB(alpha = 0.1, class_prior = None, fit_prior =None)
			self.nb = self.nb_classifier.fit(self.X_train, self.y_train)
			self.nb_pred = self.nb.predict(self.X_test)
			self.nb_accuracy = accuracy_score(self.y_test, self.nb_pred)


			# K-nearest neighbors
			self.knn_classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p=2)
			self.knn = self.knn_classifier.fit(self.X_train,self.y_train)
			self.knn_pred = self.knn.predict(self.X_test)
			self.knn_accuracy = accuracy_score(self.y_test, self.knn_pred)

			# Support vector machines
			self.svm_classifier = SVC()
			self.svm =self.svm_classifier.fit(self.X_train,self.y_train)
			self.svm_pred = self.svm.predict(self.X_test)
			self.svm_accuracy = accuracy_score(self.y_test, self.svm_pred)

			classifier_name = ['LogisticRegression', 'MultinomialNB','KNeighborsClassifier (nn=5)', 'SVC']
			accuracies  = [self.logist_accuracy, self.nb_accuracy, self.knn_accuracy, self.svm_accuracy]

			accuracy = pd.DataFrame({"Sklearn Classifier": classifier_name, "Accuracy Score": accuracies})
			print(accuracy.sort_values(by = ['Accuracy Score'], ascending = False))

		else: 
			print("Pending: add options for regression!")






