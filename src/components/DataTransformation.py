from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class ColumnTransformWithSplit:
    def __init__(self):
        pass
    
    
    def column_transformer(self, num_features, cat_features):
        num_pipeline = Pipeline([('imputer',SimpleImputer(missing_values=np.nan, strategy='mean')),
                                      ('scaling',StandardScaler())])
        cat_pipeline = Pipeline([('imputer',SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
                                      ('one_hot_encoder',OneHotEncoder())])
        preprocessor = ColumnTransformer([
            ("cat_pipeline", cat_pipeline, cat_features),
            ("num_pipeline", num_pipeline, num_features),
        ])
        return preprocessor
    
    
    def label_encoder(self):
        label_encoder = LabelEncoder()
        return label_encoder
    
    
    def get(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,shuffle=True)
        return X_train, X_test, y_train, y_test


    