import pandas as pd
import warnings
import os
from src.helper import load_pkl_file
warnings.filterwarnings('ignore')

def clean_dataset(dataset_path):
    data = pd.read_csv(f'{dataset_path}')
    if 'Unnamed: 0' in data.columns:
        data.drop(columns='Unnamed: 0', inplace=True)
        return data
    return data


class Data:
    def __init__(self, dataset_path, target='Target'):
        self.data = clean_dataset(dataset_path) 
        self.X = self.data.drop(columns=target)
        self.y = self.data[target].to_numpy()
        self.target = target
    
    def get_columns(self):
        return self.data.columns        
    
    def info(self):
        return self.data.info()
    
    def columns_split(self):
        numeric_features = [feature for feature in self.X.columns if self.X[feature].dtype != "O"]
        categorical_features = [feature for feature in self.X.columns if self.X[feature].dtype == "O"]
        return numeric_features, categorical_features
    
    def columns_split_for_display(self):
        numeric_features = [feature for feature in self.data.columns if self.data[feature].dtype != "O"]
        categorical_features = [feature for feature in self.data.columns if self.data[feature].dtype == "O"]
        return numeric_features, categorical_features
        
    def dataprocessing(self, cts, ctpath, lepath):
        y_encoder = None
        if self.y.dtype == "O":
            if not os.path.exists(lepath):
                    y_encoder = cts.label_encoder()
                    self.y = y_encoder.fit_transform(self.y)
            else:
                y_encoder = load_pkl_file(lepath)
                self.y = y_encoder.transform(self.y)

        X_train, X_test, y_train, y_test = cts.get(self.X, self.y)
        numeric_features, categorical_features = self.columns_split()
        
        if not os.path.exists(ctpath):
            X_preprocessor = cts.column_transformer(numeric_features, categorical_features)
            X_train_transformed_data = X_preprocessor.fit_transform(X_train)
            X_test_transformed_data = X_preprocessor.transform(X_test)
        else:
            X_preprocessor = load_pkl_file(ctpath)
            X_train_transformed_data = X_preprocessor.transform(X_train)
            X_test_transformed_data = X_preprocessor.transform(X_test)

        
        return X_train_transformed_data, X_test_transformed_data, y_train, y_test, X_preprocessor, y_encoder

        
        
