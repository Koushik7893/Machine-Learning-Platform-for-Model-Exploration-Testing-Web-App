from src.components.DataIngestion import Data
from src.pipelines.Models import ModelIntializer
from src.pipelines.ModelParams import ParamsInit
from src.components.DataTransformation import ColumnTransformWithSplit
from src.helper import datasets_path, save_pkl_file, load_pkl_file, compute_classification_results, compute_regression_results, all_results_json
import os
columntransformsplit = ColumnTransformWithSplit()
mod_int = ModelIntializer()
params_init = ParamsInit()

class ModelTraining:
    def __init__(self):
        self.datasets_info = datasets_path()
        
    def compute_results(self, typetotrain, y_true, y_pred):
        if typetotrain == 'classification':
            model_res = compute_classification_results(y_pred, y_true)
        else:
            model_res = compute_regression_results(y_pred, y_true)
        return model_res
    
    def TrainAllDatasets(self):
        results = {}
        for typetotrain in list(self.datasets_info.keys()):    ## Classification, regression, etc.
            datasets_res = {}
            model_names = mod_int.get(typetotrain)    ## Get all Models names
            datasets = self.datasets_info[typetotrain]    ## Get all Datasets names, paths

            for dataset_name, dataset_path in datasets.items():    ## Datasets names, paths 
                model_results = {}
                column_transformer_path = f'artifacts/{typetotrain}/{dataset_name}/{dataset_name}_preprocessor.pkl'
                label_encoder_path = f'artifacts/{typetotrain}/{dataset_name}/{dataset_name}_encoder.pkl'
                
                data = Data(dataset_path)
                x_train, x_test, y_train, y_test, preprocessor, y_encoder = data.dataprocessing(columntransformsplit, column_transformer_path,label_encoder_path)
                
                if not os.path.exists(column_transformer_path):
                        save_pkl_file(column_transformer_path, preprocessor)
                if not os.path.exists(label_encoder_path) and y_encoder != None:
                        save_pkl_file(label_encoder_path, y_encoder)
                
                for model_name in model_names:
                    model_path = f'artifacts/{typetotrain}/{dataset_name}/models/{model_name}.pkl'
                    
                    if os.path.exists(model_path):
                        model = load_pkl_file(model_path)      
                    else:                 
                        model = mod_int.model_intializer(model_name, typetotrain)()
                        model.fit(x_train, y_train)
                        save_pkl_file(model_path, model)
                        
                    y_pred = model.predict(x_test)
                    model_result = self.compute_results(typetotrain, y_test, y_pred)
                    model_results[model_name] = model_result
                datasets_res[dataset_name] = model_results
            results[typetotrain] = datasets_res
        all_results_json(results)
    

class CategoryTrainer:                       
    def __init__(self,  category, dataset_path,dataset_name):
        self.category = category
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
 
    def compute_results(self, typetotrain, y_true, y_pred):
        if typetotrain == 'classification':
            model_res = compute_classification_results(y_pred, y_true)
        else:
            model_res = compute_regression_results(y_pred, y_true)
        return model_res
    
    def get_data(self):
        column_transformer_path = f'artifacts/{self.category}/{self.dataset_name}/{self.dataset_name}_preprocessor.pkl'
        label_encoder_path = f'artifacts/{self.category}/{self.dataset_name}/{self.dataset_name}_encoder.pkl'
        data = Data(self.dataset_path)
        self.dataframe = data.data
        self.x_train, self.x_test, self.y_train, self.y_test, _, _ = data.dataprocessing(columntransformsplit, column_transformer_path,label_encoder_path)

    
    def trainer(self, params, model):
        model.set_params(**params)
        model.fit(self.x_train, self.y_train)
        y_pred = model.predict(self.x_test)
        model_result = self.compute_results(self.category, self.y_test, y_pred)
        return model_result, y_pred



