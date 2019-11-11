from sklearn import svm  # Classifier SVN
from sklearn.datasets import load_svmlight_file, load_svmlight_files
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import numpy
from BayesianOptimization import BayesianOptimization



dir_train = [
    "/home/claudiovaliense/projetos/metalazy2/metalazy/metalazy/example/data/stanford_tweets_tfIdf_5fold/train0",
    "/home/claudiovaliense/projetos/metalazy2/metalazy/metalazy/example/data/stanford_tweets_tfIdf_5fold/train1",
    "/home/claudiovaliense/projetos/metalazy2/metalazy/metalazy/example/data/stanford_tweets_tfIdf_5fold/train2",
    "/home/claudiovaliense/projetos/metalazy2/metalazy/metalazy/example/data/stanford_tweets_tfIdf_5fold/train3",
    "/home/claudiovaliense/projetos/metalazy2/metalazy/metalazy/example/data/stanford_tweets_tfIdf_5fold/train4"]

dir_test = ["/home/claudiovaliense/projetos/metalazy2/metalazy/metalazy/example/data/stanford_tweets_tfIdf_5fold/test0",
            "/home/claudiovaliense/projetos/metalazy2/metalazy/metalazy/example/data/stanford_tweets_tfIdf_5fold/test1",
            "/home/claudiovaliense/projetos/metalazy2/metalazy/metalazy/example/data/stanford_tweets_tfIdf_5fold/test2",
            "/home/claudiovaliense/projetos/metalazy2/metalazy/metalazy/example/data/stanford_tweets_tfIdf_5fold/test3",
            "/home/claudiovaliense/projetos/metalazy2/metalazy/metalazy/example/data/stanford_tweets_tfIdf_5fold/test4"]

f1_micro=[]
f1_macro=[]
for index_file in range(5):
    model_svm = svm.SVC()
    model_svm.__init__(**{'kernel': 'linear', 'C': 1, 'verbose': False, 'probability': False,
                    'degree': 3, 'shrinking': True,
                    'decision_function_shape': None, 'random_state': None,
                    'tol': 0.001, 'cache_size': 25000, 'coef0': 0.0, 'gamma': 'auto',
                    'class_weight': None, 'random_state': 42})

    x_train, y_train, x_test, y_test = load_svmlight_files([open(dir_train[index_file], 'rb'), open(dir_test[index_file], 'rb')])
    
    C = numpy.append(2.0 ** numpy.arange(-5, 15, 2), 1) 
    tuned_parameters_svm = [{'C':C}]
    bayer = BayesianOptimization(model_svm, x_train, y_train)
    print('Best param: ', bayer.fit(tuned_parameters_svm))


    # best param svm
    '''grid_svm = GridSearchCV(model_svm, tuned_parameters_svm,  cv=3, scoring='f1_macro')
    grid_svm.fit(x_train, y_train)
    best_param_svm = grid_svm.best_params_
    #print('best param: ', best_param_svm)
    y_pred = grid_svm.predict(x_test)'''        

    model_svm.fit(x_train, y_train) # sem best param and cro
    y_pred = model_svm.predict(x_test) # sem best param
    f1_macro.append(f1_score(y_test, y_pred, average='macro'))
    f1_micro.append(f1_score(y_test, y_pred, average='micro'))

media_f1_macro = sum(f1_macro)/5
media_f1_micro = sum(f1_micro)/5
# desvio padrão
sum_f1_macro=0
sum_f1_micro=0
for index in range(5):
    sum_f1_macro += (f1_macro[index]-media_f1_macro)**(2)
    sum_f1_micro += (f1_micro[index] - media_f1_micro) ** (2)

print(f1_macro)
print('Média Macro F1: ', media_f1_macro)
print("Desvio padrão Macro F1: ", (sum_f1_macro/5)**(1/2))
print('Média Micro F1:  ', media_f1_micro)
print("Desvio padrão Micro F1: ", (sum_f1_micro/5)**(1/2))

'''
print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))
precision, recall, fscore, support = score(y_test, y_pred)
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))
'''
