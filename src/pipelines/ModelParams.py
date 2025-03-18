from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut
import optuna
from tpot import TPOTClassifier
# import autosklearn.classification
import numpy as np

available_params_1 = {
    'LogisticRegression': {
        'penalty': 'str, default="l2". Regularization type (l1, l2, elasticnet).',
        'dual': 'bool, default=False. Dual or primal formulation.',
        'C': 'float, default=1.0. Inverse of regularization strength.',
        'fit_intercept': 'bool, default=True. Whether to calculate the intercept.',
        'solver': 'str, default="lbfgs". Algorithm to use in the optimization problem.',
        'max_iter': 'int, default=100. Maximum number of iterations.',
        'n_jobs': 'int, default=None. Number of parallel jobs to run.'
    },
    'SVC': {
        'C': 'float, default=1.0. Regularization parameter.',
        'kernel': 'str, default="rbf". Kernel type (linear, poly, rbf, sigmoid).',
        'degree': 'int, default=3. Degree of the polynomial kernel function.',
        'gamma': 'str or float, default="scale". Kernel coefficient.',
        'probability': 'bool, default=False. Whether to enable probability estimates.',
    },
    'GaussianNB': {
        'var_smoothing': 'float, default=1e-9. Portion of the largest variance of all features added to variances.',
    },
    'KNeighborsClassifier': {
        'n_neighbors': 'int, default=5. Number of neighbors to use.',
        'weights': 'str, default="uniform". Weight function ("uniform", "distance").',
        'algorithm': 'str, default="auto". Algorithm used to compute nearest neighbors.',
        'leaf_size': 'int, default=30. Leaf size for BallTree or KDTree.',
    },
    'DecisionTreeClassifier': {
        'criterion': 'str, default="gini". Function to measure split quality ("gini", "entropy").',
        'splitter': 'str, default="best". Strategy to split nodes ("best", "random").',
        'max_depth': 'int, default=None. Maximum depth of the tree.',
        'min_samples_split': 'int, default=2. Minimum number of samples required to split an internal node.',
    },
    'RandomForestClassifier': {
        'n_estimators': 'int, default=100. Number of trees in the forest.',
        'criterion': 'str, default="gini". Function to measure split quality.',
        'max_depth': 'int, default=None. Maximum depth of the trees.',
        'min_samples_split': 'int, default=2. Minimum number of samples required to split an internal node.',
        'n_jobs': 'int, default=None. Number of parallel jobs to run.',
    },
    'GradientBoostingClassifier': {
        'learning_rate': 'float, default=0.1. Learning rate shrinks the contribution of each tree.',
        'n_estimators': 'int, default=100. Number of boosting stages.',
        'max_depth': 'int, default=3. Maximum depth of individual regression estimators.',
        'subsample': 'float, default=1.0. Fraction of samples used for fitting the individual base learners.',
    },
    'AdaBoostClassifier': {
        'n_estimators': 'int, default=50. Number of weak learners (trees).',
        'learning_rate': 'float, default=1.0. Weight applied to each classifier.',
    },
    'XGBClassifier': {
        'n_estimators': 'int, default=100. Number of boosting rounds.',
        'max_depth': 'int, default=6. Maximum depth of a tree.',
        'learning_rate': 'float, default=0.3. Boosting learning rate.',
        'subsample': 'float, default=1.0. Fraction of samples used for training.',
    },
    'LinearRegression': {
            'fit_intercept': 'bool, default=True. Whether to calculate the intercept.',
            'normalize': 'bool, default=False. Deprecated; used to normalize before regression.',
            'copy_X': 'bool, default=True. If True, X will be copied; else, overwritten.',
            'n_jobs': 'int, default=None. Number of parallel jobs to run.'
    },
    'Lasso': {
        'alpha': 'float, default=1.0. Constant that multiplies the L1 term.',
        'max_iter': 'int, default=1000. Maximum number of iterations.',
        'tol': 'float, default=0.0001. Tolerance for the optimization.',
    },
    'Ridge': {
        'alpha': 'float, default=1.0. Regularization strength.',
        'solver': 'str, default="auto". Solver to use in the optimization problem.',
    },
    'ElasticNet': {
        'alpha': 'float, default=1.0. Constant that multiplies the penalty terms.',
        'l1_ratio': 'float, default=0.5. The mix ratio between L1 and L2 regularization.',
        'max_iter': 'int, default=1000. Maximum number of iterations.',
    },
    'SVR': {
        'kernel': 'str, default="rbf". Specifies the kernel type to be used.',
        'degree': 'int, default=3. Degree of the polynomial kernel.',
        'C': 'float, default=1.0. Regularization parameter.',
        'epsilon': 'float, default=0.1. Epsilon-tube within which no penalty is given.',
    },
    'KNeighborsRegressor': {
        'n_neighbors': 'int, default=5. Number of neighbors to use.',
        'weights': 'str, default="uniform". Weight function ("uniform", "distance").',
        'algorithm': 'str, default="auto". Algorithm used to compute nearest neighbors.',
    },
    'DecisionTreeRegressor': {
        'criterion': 'str, default="mse". Function to measure split quality ("mse", "mae").',
        'splitter': 'str, default="best". Strategy to split nodes.',
        'max_depth': 'int, default=None. Maximum depth of the tree.',
        'min_samples_split': 'int, default=2. Minimum number of samples required to split an internal node.',
    },
    'RandomForestRegressor': {
        'n_estimators': 'int, default=100. Number of trees in the forest.',
        'criterion': 'str, default="mse". Function to measure split quality.',
        'max_depth': 'int, default=None. Maximum depth of the trees.',
        'min_samples_split': 'int, default=2. Minimum number of samples required to split an internal node.',
        'n_jobs': 'int, default=None. Number of parallel jobs to run.'
    },
    'GradientBoostingRegressor': {
        'learning_rate': 'float, default=0.1. Shrinks contribution of each tree.',
        'n_estimators': 'int, default=100. Number of boosting stages.',
        'max_depth': 'int, default=3. Maximum depth of the individual estimators.',
        'subsample': 'float, default=1.0. Fraction of samples used for fitting the individual base learners.',
    },
    'AdaBoostRegressor': {
        'n_estimators': 'int, default=50. Number of weak learners.',
        'learning_rate': 'float, default=1.0. Weight applied to each regressor.',
    },
    'XGBRegressor': {
        'n_estimators': 'int, default=100. Number of boosting rounds.',
        'max_depth': 'int, default=6. Maximum depth of a tree.',
        'learning_rate': 'float, default=0.3. Boosting learning rate.',
        'subsample': 'float, default=1.0. Fraction of samples used for training.',
    }
}
available_params = {
    'LogisticRegression': [
    {'Parameter': 'penalty', 'Dtype': 'str', 'Default Value': 'l2', 'Description': 'Regularization type (l1, l2, elasticnet).'},
    {'Parameter': 'dual', 'Dtype': 'bool', 'Default Value': 'False', 'Description': 'Dual or primal formulation.'},
    {'Parameter': 'C', 'Dtype': 'float', 'Default Value': '1.0', 'Description': 'Inverse of regularization strength.'},
    {'Parameter': 'fit_intercept', 'Dtype': 'bool', 'Default Value': 'True', 'Description': 'Whether to calculate the intercept.'},
    {'Parameter': 'solver', 'Dtype': 'str', 'Default Value': 'lbfgs', 'Description': 'Algorithm to use in the optimization problem.'},
    {'Parameter': 'max_iter', 'Dtype': 'int', 'Default Value': '100', 'Description': 'Maximum number of iterations.'},
    {'Parameter': 'n_jobs', 'Dtype': 'int', 'Default Value': 'None', 'Description': 'Number of parallel jobs to run.'}
    ],
    'SVC': [
    {'Parameter': 'C', 'Dtype': 'float', 'Default Value': '1.0', 'Description': 'Regularization parameter.'},
    {'Parameter': 'kernel', 'Dtype': 'str', 'Default Value': 'rbf', 'Description': 'Kernel type (linear, poly, rbf, sigmoid).'},
    {'Parameter': 'degree', 'Dtype': 'int', 'Default Value': '3', 'Description': 'Degree of the polynomial kernel function.'},
    {'Parameter': 'gamma', 'Dtype': 'str or float', 'Default Value': 'scale', 'Description': 'Kernel coefficient.'},
    {'Parameter': 'probability', 'Dtype': 'bool', 'Default Value': 'False', 'Description': 'Whether to enable probability estimates.'}
    ],
    'GaussianNB': [
        {'Parameter': 'var_smoothing', 'Dtype': 'float', 'Default Value': '1e-9', 'Description': 'Portion of the largest variance of all features added to variances.'}
    ],
    'KNeighborsClassifier': [
    {'Parameter': 'n_neighbors', 'Dtype': 'int', 'Default Value': '5', 'Description': 'Number of neighbors to use.'},
    {'Parameter': 'weights', 'Dtype': 'str', 'Default Value': 'uniform', 'Description': 'Weight function ("uniform", "distance").'},
    {'Parameter': 'algorithm', 'Dtype': 'str', 'Default Value': 'auto', 'Description': 'Algorithm used to compute nearest neighbors.'},
    {'Parameter': 'leaf_size', 'Dtype': 'int', 'Default Value': '30', 'Description': 'Leaf size for BallTree or KDTree.'}
    ],
    'DecisionTreeClassifier': [
        {'Parameter': 'criterion', 'Dtype': 'str', 'Default Value': 'gini', 'Description': 'Function to measure split quality ("gini", "entropy").'},
        {'Parameter': 'splitter', 'Dtype': 'str', 'Default Value': 'best', 'Description': 'Strategy to split nodes ("best", "random").'},
        {'Parameter': 'max_depth', 'Dtype': 'int', 'Default Value': 'None', 'Description': 'Maximum depth of the tree.'},
        {'Parameter': 'min_samples_split', 'Dtype': 'int', 'Default Value': '2', 'Description': 'Minimum number of samples required to split an internal node.'}
    ],'RandomForestClassifier': [
        {'Parameter': 'n_estimators', 'Dtype': 'int', 'Default Value': '100', 'Description': 'Number of trees in the forest.'},
        {'Parameter': 'criterion', 'Dtype': 'str', 'Default Value': 'gini', 'Description': 'Function to measure split quality.'},
        {'Parameter': 'max_depth', 'Dtype': 'int', 'Default Value': 'None', 'Description': 'Maximum depth of the trees.'},
        {'Parameter': 'min_samples_split', 'Dtype': 'int', 'Default Value': '2', 'Description': 'Minimum number of samples required to split an internal node.'},
        {'Parameter': 'n_jobs', 'Dtype': 'int', 'Default Value': 'None', 'Description': 'Number of parallel jobs to run.'}
    ],
    'GradientBoostingClassifier': [
        {'Parameter': 'learning_rate', 'Dtype': 'float', 'Default Value': '0.1', 'Description': 'Learning rate shrinks the contribution of each tree.'},
        {'Parameter': 'n_estimators', 'Dtype': 'int', 'Default Value': '100', 'Description': 'Number of boosting stages.'},
        {'Parameter': 'max_depth', 'Dtype': 'int', 'Default Value': '3', 'Description': 'Maximum depth of individual regression estimators.'},
        {'Parameter': 'subsample', 'Dtype': 'float', 'Default Value': '1.0', 'Description': 'Fraction of samples used for fitting the individual base learners.'}
    ],
    'AdaBoostClassifier': [
    {'Parameter': 'n_estimators', 'Dtype': 'int', 'Default Value': '50', 'Description': 'Number of weak learners (trees).'},
    {'Parameter': 'learning_rate', 'Dtype': 'float', 'Default Value': '1.0', 'Description': 'Weight applied to each classifier.'}
    ],
    'XGBClassifier': [
        {'Parameter': 'n_estimators', 'Dtype': 'int', 'Default Value': '100', 'Description': 'Number of boosting rounds.'},
        {'Parameter': 'max_depth', 'Dtype': 'int', 'Default Value': '6', 'Description': 'Maximum depth of a tree.'},
        {'Parameter': 'learning_rate', 'Dtype': 'float', 'Default Value': '0.3', 'Description': 'Boosting learning rate.'},
        {'Parameter': 'subsample', 'Dtype': 'float', 'Default Value': '1.0', 'Description': 'Fraction of samples used for training.'}
    ],
    'LinearRegression': [
        {'Parameter': 'fit_intercept', 'Dtype': 'bool', 'Default Value': 'True', 'Description': 'Whether to calculate the intercept.'},
        {'Parameter': 'normalize', 'Dtype': 'bool', 'Default Value': 'False', 'Description': 'Deprecated; used to normalize before regression.'},
        {'Parameter': 'copy_X', 'Dtype': 'bool', 'Default Value': 'True', 'Description': 'If True, X will be copied; else, overwritten.'},
        {'Parameter': 'n_jobs', 'Dtype': 'int', 'Default Value': 'None', 'Description': 'Number of parallel jobs to run.'}
    ],
    'Lasso': [
        {'Parameter': 'alpha', 'Dtype': 'float', 'Default Value': '1.0', 'Description': 'Constant that multiplies the L1 term.'},
        {'Parameter': 'max_iter', 'Dtype': 'int', 'Default Value': '1000', 'Description': 'Maximum number of iterations.'},
        {'Parameter': 'tol', 'Dtype': 'float', 'Default Value': '0.0001', 'Description': 'Tolerance for the optimization.'}
    ],
    'Ridge': [
        {'Parameter': 'alpha', 'Dtype': 'float', 'Default Value': '1.0', 'Description': 'Regularization strength.'},
        {'Parameter': 'solver', 'Dtype': 'str', 'Default Value': 'auto', 'Description': 'Solver to use in the optimization problem.'}
    ],
    'ElasticNet': [
        {'Parameter': 'alpha', 'Dtype': 'float', 'Default Value': '1.0', 'Description': 'Constant that multiplies the penalty terms.'},
        {'Parameter': 'l1_ratio', 'Dtype': 'float', 'Default Value': '0.5', 'Description': 'The mix ratio between L1 and L2 regularization.'},
        {'Parameter': 'max_iter', 'Dtype': 'int', 'Default Value': '1000', 'Description': 'Maximum number of iterations.'}
    ],
    'SVR': [
        {'Parameter': 'kernel', 'Dtype': 'str', 'Default Value': 'rbf', 'Description': 'Specifies the kernel type to be used.'},
        {'Parameter': 'degree', 'Dtype': 'int', 'Default Value': '3', 'Description': 'Degree of the polynomial kernel.'},
        {'Parameter': 'C', 'Dtype': 'float', 'Default Value': '1.0', 'Description': 'Regularization parameter.'},
        {'Parameter': 'epsilon', 'Dtype': 'float', 'Default Value': '0.1', 'Description': 'Epsilon-tube within which no penalty is given.'}
    ],
    'KNeighborsRegressor': [
        {'Parameter': 'n_neighbors', 'Dtype': 'int', 'Default Value': '5', 'Description': 'Number of neighbors to use.'},
        {'Parameter': 'weights', 'Dtype': 'str', 'Default Value': 'uniform', 'Description': 'Weight function ("uniform", "distance").'},
        {'Parameter': 'algorithm', 'Dtype': 'str', 'Default Value': 'auto', 'Description': 'Algorithm used to compute nearest neighbors.'}
    ],
    'DecisionTreeRegressor': [
        {'Parameter': 'criterion', 'Dtype': 'str', 'Default Value': 'mse', 'Description': 'Function to measure split quality ("mse", "mae").'},
        {'Parameter': 'splitter', 'Dtype': 'str', 'Default Value': 'best', 'Description': 'Strategy to split nodes.'},
        {'Parameter': 'max_depth', 'Dtype': 'int', 'Default Value': 'None', 'Description': 'Maximum depth of the tree.'},
        {'Parameter': 'min_samples_split', 'Dtype': 'int', 'Default Value': '2', 'Description': 'Minimum number of samples required to split an internal node.'}
    ],
    'RandomForestRegressor': [
        {'Parameter': 'n_estimators', 'Dtype': 'int', 'Default Value': '100', 'Description': 'Number of trees in the forest.'},
        {'Parameter': 'criterion', 'Dtype': 'str', 'Default Value': 'mse', 'Description': 'Function to measure split quality.'},
        {'Parameter': 'max_depth', 'Dtype': 'int', 'Default Value': 'None', 'Description': 'Maximum depth of the trees.'},
        {'Parameter': 'min_samples_split', 'Dtype': 'int', 'Default Value': '2', 'Description': 'Minimum number of samples required to split an internal node.'},
        {'Parameter': 'n_jobs', 'Dtype': 'int', 'Default Value': 'None', 'Description': 'Number of parallel jobs to run.'}
    ],
    'GradientBoostingRegressor': [
        {'Parameter': 'learning_rate', 'Dtype': 'float', 'Default Value': '0.1', 'Description': 'Shrinks contribution of each tree.'},
        {'Parameter': 'n_estimators', 'Dtype': 'int', 'Default Value': '100', 'Description': 'Number of boosting stages.'},
        {'Parameter': 'max_depth', 'Dtype': 'int', 'Default Value': '3', 'Description': 'Maximum depth of the individual estimators.'},
        {'Parameter': 'subsample', 'Dtype': 'float', 'Default Value': '1.0', 'Description': 'Fraction of samples used for fitting the individual base learners.'}
    ],
    'AdaBoostRegressor': [
        {'Parameter': 'n_estimators', 'Dtype': 'int', 'Default Value': '50', 'Description': 'Number of weak learners.'},
        {'Parameter': 'learning_rate', 'Dtype': 'float', 'Default Value': '1.0', 'Description': 'Weight applied to each regressor.'}
    ],
    'XGBRegressor': [
        {'Parameter': 'n_estimators', 'Dtype': 'int', 'Default Value': '100', 'Description': 'Number of boosting rounds.'},
        {'Parameter': 'max_depth', 'Dtype': 'int', 'Default Value': '6', 'Description': 'Maximum depth of a tree.'},
        {'Parameter': 'learning_rate', 'Dtype': 'float', 'Default Value': '0.3', 'Description': 'Boosting learning rate.'},
        {'Parameter': 'subsample', 'Dtype': 'float', 'Default Value': '1.0', 'Description': 'Fraction of samples used for training.'}
    ]
}
params = {
        'LogisticRegression' : {          
            'solver': [0.1, 0.5, 1.0, 10.0], 
            'penalty': ['l1', 'l2', 'elasticnet'], 
            'C': ['lbfgs', 'liblinear', 'saga', 'newton-cg', 'sag'], 
            'max_iter': [100, 200, 300],  
        },
        'SVC': {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['linear','poly','rbf'],
            'degree': [2, 3, 4],
            'gamma': ['scale', 'auto'],
        },
        'GaussianNB': {
            'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
        },
        'KNeighborsClassifier': {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'leaf_size': [20, 30, 40, 50]
        },
        'DecisionTreeClassifier': {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'max_depth': [None, 5, 10, 15,30],
            'min_samples_split': [2, 5, 10]
        },
        'RandomForestClassifier': {
            'n_estimators': [50, 100, 200],
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
        },
        'GradientBoostingClassifier': {
            'learning_rate': [0.01, 0.1, 0.2],
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'subsample': [0.5, 0.75, 1.0]
        },
        'AdaBoostClassifier': {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.5, 1.0, 1.5]
        },
        'XGBClassifier': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.5, 0.75, 1.0]
    },
    
        'LinearRegression': {
            'fit_intercept': [True, False],
            'n_jobs': [-1]
        },
        'Lasso': {
            'alpha': [0.1, 1.0, 10.0],
            'max_iter': [100, 500, 1000],
            'tol': [0.0001, 0.001, 0.01]
        },
        'Ridge': {
            'alpha': [0.1, 1.0, 10.0],
            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sag', 'saga'],
            'tol': [0.0001, 0.001, 0.01]
        },
        'ElasticNet': {
            'alpha': [0.1, 1.0, 10.0],
            'l1_ratio': [0.1, 0.5, 0.9],
            'max_iter': [100, 500, 1000]
        },
        'SVR': {
            'kernel': ['poly', 'rbf'],
            'degree': [2, 3, 4],
            'C': [0.1, 1.0, 10.0],
            'epsilon': [0.01, 0.1, 0.5]
        },
        'KNeighborsRegressor': {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        },
        'DecisionTreeRegressor': {
            'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],
            'splitter': ['best', 'random'],
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10]
        },
        'RandomForestRegressor': {
            'n_estimators': [50, 100, 200],
            'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
        },
        'GradientBoostingRegressor': {
            'learning_rate': [0.01, 0.1, 0.2],
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'subsample': [0.5, 0.75, 1.0]
        },
        'AdaBoostRegressor': {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.5, 1.0, 1.5]
        },
        'XGBRegressor': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.5, 0.75, 1.0]
    }
}


class ParamsInit:
    def __init__(self):
        self.params = params
    
    def Grid_SearchCV(self,typetotrain, model, model_params):
        if typetotrain == "classification":
            search = GridSearchCV(estimator=model, param_grid=model_params, cv=5, scoring = 'accuracy', n_jobs=-1)
        else:
            search = GridSearchCV(estimator=model, param_grid=model_params, cv=5, scoring = 'neg_mean_squared_error', n_jobs=-1)
        return search
    
    def Randomized_SearchCV(self,typetotrain, model, model_params):
        count = 0
        for i in model_params.values():
            count += len(i)
        n_iter = np.round(count * 3 / 4)
        if typetotrain == "classification":
            search = RandomizedSearchCV(estimator=model, param_grid=model_params, cv=5, scoring = 'accuracy', n_iter=n_iter, n_jobs=-1)
        else:
            search = RandomizedSearchCV(estimator=model, param_grid=model_params, cv=5, scoring = 'neg_mean_squared_error', n_iter=n_iter, n_jobs=-1)
        return search
    
    def Optuna_tuning(self, objective, n_trials):
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        
        return study
    
    
    def tpot_tuner(self):
        tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2, random_state=42)
        return tpot