from sklearn import svm  # Classifier SVN
from sklearn.datasets import load_svmlight_file, load_svmlight_files
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import numpy
import timeit  # Measure time
from hyperopt import hp
import sys # Import other directory
sys.path.append("../../doutorado") 
import claudio_funcoes as cv  # Functions utils author

"""Example execution: python3.6 stanford_tweets linear"""

name_dataset=sys.argv[1]
kernel=sys.argv[2]

ini = timeit.default_timer()
dir_train = [    
    "/home/claudiovaliense/dataset/" + name_dataset +"/representations/5-folds/TFIDF_removed_stopwords_mindf1/extract/train/train0",
    "/home/claudiovaliense/dataset/" + name_dataset +"/representations/5-folds/TFIDF_removed_stopwords_mindf1/extract/train/train1",
    "/home/claudiovaliense/dataset/" + name_dataset +"/representations/5-folds/TFIDF_removed_stopwords_mindf1/extract/train/train2",
    "/home/claudiovaliense/dataset/" + name_dataset +"/representations/5-folds/TFIDF_removed_stopwords_mindf1/extract/train/train3",
    "/home/claudiovaliense/dataset/" + name_dataset +"/representations/5-folds/TFIDF_removed_stopwords_mindf1/extract/train/train4"]

dir_test = [
    "/home/claudiovaliense/dataset/" + name_dataset +"/representations/5-folds/TFIDF_removed_stopwords_mindf1/extract/test/test0",
    "/home/claudiovaliense/dataset/" + name_dataset +"/representations/5-folds/TFIDF_removed_stopwords_mindf1/extract/test/test1",
    "/home/claudiovaliense/dataset/" + name_dataset +"/representations/5-folds/TFIDF_removed_stopwords_mindf1/extract/test/test2",
    "/home/claudiovaliense/dataset/" + name_dataset +"/representations/5-folds/TFIDF_removed_stopwords_mindf1/extract/test/test3",
    "/home/claudiovaliense/dataset/" + name_dataset +"/representations/5-folds/TFIDF_removed_stopwords_mindf1/extract/test/test4"]

y_test_folds = []
y_pred_folds = []
best_param_folds=[]
tunned_param_lbd = { 'kernel': 'rbf', 'verbose': False, 'probability': False,
                    'degree': 3, 'shrinking': True,
                    'decision_function_shape': None, 
                    'tol': 0.001, 'cache_size': 25000, 'coef0': 0.0, 'gamma': 'auto',
                    'class_weight': None, 'random_state': 42}

C = numpy.append(2.0 ** numpy.arange(-5, 15, 2), 1)    
#tuned_parameters = [{'kernel': ['linear'], 'C': [1,2, 3,4,5,10, 100, 1000, 1500,5000]}, 
#							{'kernel': ['rbf'], 'gamma': [2.5, 2, 1.5, 1, 1e-1, 1e-2, 1e-3, 1e-4],
#			       			     'C': [1, 10, 100, 1000, 1500,5000, 10000]}]

#tuned_parameters = [#{'kernel': ['linear'], 'C': C}, 
#                   {'kernel': ['rbf', 'linear'], 'gamma': [2.5, 2, 1.5, 1, 1e-1, 1e-2, 1e-3, 1e-4, 'auto'], 'C': C}]

C_range = C
gamma_range = numpy.logspace(-9, 3, 13)
param_grid = dict(gamma=['auto'], C=C_range)

#tuned_parameters =[{'kernel': ['rbf'], 'C': C, 'gamma':['auto']},
 #                  {'kernel':['linear'], 'C': C},
  #                 {'kernel': ['rbf'], 'C': C, 'gamma': gamma_range }]


#C = [0.0001, 0.001, 0.01, 0.1, 1, 2, 3, 5, 10, 20, 100, 1000, 10000,100000]
tuned_parameters_svm = [{'C': C}]
    
for index_file in range(5):
    model_svm = svm.SVC()    
    model_svm.__init__(**tunned_param_lbd)    
    x_train, y_train, x_test, y_test = load_svmlight_files([open(dir_train[index_file], 'rb'), open(dir_test[index_file], 'rb')])
    


    # best param svm grid 
    grid_svm = GridSearchCV(model_svm, param_grid=param_grid,  cv=3, scoring='f1_micro', n_jobs=1)        
    grid_svm.fit(x_train, y_train)   
    print(grid_svm.get_params() )
    best_param_svm = grid_svm.best_params_  
    best_param_folds.append(best_param_svm)              
    y_pred = grid_svm.predict(x_test)
    #---------    
    
    y_test_folds.append(y_test)    
    y_pred_folds.append(y_pred)                

print(tunned_param_lbd)
cv.statistics_experiment(y_test_folds, y_pred_folds, best_param_folds)
print("Time End: %f" % (timeit.default_timer() - ini))


'''
#print('Shape', x_train.shape)

print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))
precision, recall, fscore, support = score(y_test, y_pred)
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))
'''
# BayesianOptimization-------
'''
#from BayesianOptimization import BayesianOptimization
kernels = ["linear"]
    hyperparameters = {
        "C": hp.uniform("C", 1, 20) #,
        #"kernel": hp.choice("kernel", kernels)
    }
    bayer = BayesianOptimization(model_svm, x_train, y_train)    
    best_param_bayesian = bayer.fit(hyperparameters)
    print('Best param: ', best_param_bayesian)
    model_svm.set_params(**best_param_bayesian)
    model_svm.fit(x_train, y_train)
    y_pred = model_svm.predict(x_test)
'''
#----------

