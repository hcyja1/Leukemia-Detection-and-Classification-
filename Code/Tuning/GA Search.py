import matplotlib.pyplot as plt
import os
from numpy import load
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Categorical, Integer, Continuous
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score
from sklearn_genetic.plots import plot_search_space
from sklearn_genetic.plots import plot_fitness_evolution
from xgboost.sklearn import XGBClassifier
from sklearn.utils.fixes import loguniform
from keras.models import load_model
import numpy as np

from sklearn import preprocessing

def loadFeatures(cnn_model,Database):
    data = load(r"../Feature Extraction/"+Database+"/Features/"+cnn_model+"_"+Database+"_features.npy")
    return data

def loadLabels(Database):
    labels = load(r"../Feature Extraction/"+Database+"/"+Database+"_labels.npy")
    return labels

def GA_tune(database):

    labels = loadLabels(database)
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    labels_encoded = le.transform(labels)

    DenseNet121 = loadFeatures("DenseNet121", database)
    VGG16 = loadFeatures("VGG16", database)

    features = np.concatenate([VGG16,DenseNet121], 1)

    X_training_features, X_test_features, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.20)
    param_grid = {'gamma': Continuous(0.00, 1.0),
                  'learning_rate': Continuous(0.05, 0.3),
                  'max_depth': Integer(2,4),
                  'colsample_bytree':  Continuous(0.5, 1.0),
                  'scale_pos_weight': Integer(1, 3),
                  'subsample':  Continuous(0.5, 1.0),
                  'n_estimators': Integer(100, 500)
                  }

    XGB_model = XGBClassifier(
        eval_metric="error",
    )


    # The main class from sklearn-genetic-opt
    evolved_estimator = GASearchCV(estimator=XGB_model,
                                  scoring='accuracy',
                                  param_grid=param_grid,
                                  n_jobs=-1,
                                  verbose=True,
                                   population_size=7,
                                   generations = 15,
                                   tournament_size=3,
                                   elitism = True,
                                   crossover_probability=0.80,
                                   mutation_probability=0.15)

    from sklearn_genetic.callbacks import ConsecutiveStopping
    consecCallback= ConsecutiveStopping(generations=3, metric='fitness')
    from sklearn_genetic.callbacks import ProgressBar
    progressCallback = ProgressBar()

    callbacks = [consecCallback,progressCallback]

    evolved_estimator.fit(X_training_features, y_train,callbacks)
    print(evolved_estimator.best_params_)
    # Use the model fitted with the best parameters
    y_predict_ga = evolved_estimator.predict(X_test_features)
    print(accuracy_score(y_test, y_predict_ga))
    plot_fitness_evolution(evolved_estimator)
    plt.show()
    plot_search_space(evolved_estimator, features=['gamma','learning_rate','scale_pos_weight', 'colsample_bytree','max_depth', 'subsample', 'n_estimators'])
    plt.show()

GA_tune("IDB1_cropped")