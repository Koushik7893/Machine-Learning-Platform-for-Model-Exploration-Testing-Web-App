import pandas as pd
import warnings
import os
from src.helper import load_pkl_file
warnings.filterwarnings('ignore')

def clean_dataset(dataset_path):
    if type(dataset_path) == str:    
        data = pd.read_csv(f'{dataset_path}')
    else:
        data = pd.read_csv(dataset_path)
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
    
    def get_x_columns(self):
        return self.X.columns  
    
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
        
    def encoder(self,cts):
            y_encoder = cts.label_encoder()
            self.y = y_encoder.fit_transform(self.y)
            return y_encoder
        
    def xencoder(self, cts, numeric_features, categorical_features, X_train, X_test):
        X_preprocessor = cts.column_transformer(numeric_features, categorical_features)
        X_train_transformed_data = X_preprocessor.fit_transform(X_train)
        X_test_transformed_data = X_preprocessor.transform(X_test)
        return X_preprocessor, X_train_transformed_data, X_test_transformed_data
    
    def split(self, cts):
        return cts.get(self.X, self.y)
    
    def dataprocessing(self, cts=None, ctpath=None, lepath=None, custom=None):
        y_encoder = None
        if self.y.dtype == "O":
            if custom != None:
                y_encoder = self.encoder(cts)
            else:
                if not os.path.exists(lepath):
                    y_encoder = self.encoder(cts)
                else:
                    y_encoder = load_pkl_file(lepath)
                    self.y = y_encoder.transform(self.y)

        X_train, X_test, y_train, y_test = cts.get(self.X, self.y)
        numeric_features, categorical_features = self.columns_split()
        if custom != None:
            X_preprocessor, X_train_transformed_data, X_test_transformed_data = self.xencoder(cts, numeric_features, categorical_features, X_train, X_test)
        else:
            if not os.path.exists(ctpath):
                X_preprocessor, X_train_transformed_data, X_test_transformed_data = self.xencoder(cts, numeric_features, categorical_features, X_train, X_test)
            else:
                X_preprocessor = load_pkl_file(ctpath)
                X_train_transformed_data = X_preprocessor.transform(X_train)
                X_test_transformed_data = X_preprocessor.transform(X_test)

        
        return X_train_transformed_data, X_test_transformed_data, y_train, y_test, X_preprocessor, y_encoder

        
        
