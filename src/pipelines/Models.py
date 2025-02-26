from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import SVC, SVR 
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBClassifier, XGBRegressor

model_dict = {'classification': 
                {'LogisticRegression': LogisticRegression,
                'SVC': SVC,
                'GaussianNB': GaussianNB,
                'KNeighborsClassifier': KNeighborsClassifier,
                'DecisionTreeClassifier': DecisionTreeClassifier,
                'RandomForestClassifier': RandomForestClassifier,
                'GradientBoostingClassifier': GradientBoostingClassifier,
                'AdaBoostClassifier': AdaBoostClassifier,
                'XGBClassifier': XGBClassifier,
                },
                
            'regression' : 
                {'LinearRegression': LinearRegression,
                'Lasso': Lasso,
                'Ridge': Ridge,
                'ElasticNet': ElasticNet,
                'SVR': SVR,
                'KNeighborsRegressor': KNeighborsRegressor,
                'DecisionTreeRegressor': DecisionTreeRegressor,
                'RandomForestRegressor': RandomForestRegressor,
                'GradientBoostingRegressor': GradientBoostingRegressor,
                'AdaBoostRegressor': AdaBoostRegressor,
                'XGBRegressor': XGBRegressor,
                }
            }
        


class ModelIntializer:
    def __init__(self):
        self.model_dict = model_dict
        
    def get(self,typetotrain):
        models = self.model_dict[typetotrain]
        return list(models.keys())
    
    def model_intializer(self, model_name, typetotrain):
        models = self.model_dict[typetotrain]
        model = models[model_name]
        return model