from src.components.DataIngestion import Data
from src.pipelines.Models import ModelIntializer
from src.pipelines.ModelParams import ParamsInit
from src.components.DataTransformation import ColumnTransformWithSplit
from sklearn.metrics import accuracy_score
from src.helper import datasets_path, save_pkl_file, load_pkl_file, compute_classification_results, compute_regression_results, all_results_json, datasets_info
import os
import time
columntransformsplit = ColumnTransformWithSplit()
mod_int = ModelIntializer()
params_init = ParamsInit()


def model_fitter(model, x_train, y_train):
    train_start_time = time.time()
    model.fit(x_train, y_train)
    train_end_time = time.time()
    return model, train_end_time - train_start_time

def test_predictor(model,x_test):
    y_pred_start_time = time.time()
    y_pred = model.predict(x_test)
    y_pred_end_time = time.time()
    return y_pred, y_pred_end_time - y_pred_start_time

def train_predictor(model,x_train):
    y_pred_train_start_time = time.time()
    y_pred_train = model.predict(x_train)
    y_pred_train_end_time = time.time()
    return y_pred_train, y_pred_train_end_time - y_pred_train_start_time

class ModelTraining:
    def __init__(self):
        self.datasets_info = datasets_path()
        
    def compute_results(self, typetotrain, y_true, y_pred, com_time):
        if typetotrain == 'classification':
            model_res = compute_classification_results(y_pred, y_true, com_time)
        else:
            model_res = compute_regression_results(y_pred, y_true, com_time)
        return model_res
    
    def TrainAllDatasets(self):
        datasets_info()
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
                x_train, x_test, y_train, y_test, preprocessor, y_encoder = data.dataprocessing(cts=columntransformsplit, ctpath=column_transformer_path,lepath=label_encoder_path)
                
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
                        model, train_time = model_fitter(model, x_train, y_train)
                        save_pkl_file(model_path, model)
                    
                    y_pred, test_inf_time = test_predictor(model,x_test)
                    test_model_result = self.compute_results(typetotrain, y_test, y_pred, test_inf_time)
                    
                    y_pred_train, train_inf_time = train_predictor(model, x_train)
                    train_model_result = self.compute_results(typetotrain, y_train, y_pred_train, train_inf_time)
                                     
                    file_size = os.path.getsize(model_path) / 1024 
                    model_result = {'train_model_result':train_model_result,
                                    'test_model_result':test_model_result,
                                    'training_time':train_time,
                                    'file_size_mb':file_size}
                    model_results[model_name] = model_result
                    
                datasets_res[dataset_name] = model_results
            results[typetotrain] = datasets_res
        all_results_json(results)
    

class CategoryTrainer:                       
    def __init__(self,  category, dataset_path, dataset_name):
        self.category = category
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
 
    def compute_results(self, typetotrain, y_true, y_pred):
        if typetotrain == 'classification':
            model_res = compute_classification_results(y_pred, y_true,0)
        else:
            model_res = compute_regression_results(y_pred, y_true,0)
        return model_res
    
    def get_data(self):
        column_transformer_path = f'artifacts/{self.category}/{self.dataset_name}/{self.dataset_name}_preprocessor.pkl'
        label_encoder_path = f'artifacts/{self.category}/{self.dataset_name}/{self.dataset_name}_encoder.pkl'
        data = Data(self.dataset_path)
        self.dataframe = data.data
        self.x_train, self.x_test, self.y_train, self.y_test, _, _ = data.dataprocessing(cts=columntransformsplit, ctpath=column_transformer_path, lepath=label_encoder_path)

    
    def trainer(self, params, model):
        model.set_params(**params)
        model.fit(self.x_train, self.y_train)
        y_pred = model.predict(self.x_test)
        model_result = self.compute_results(self.category, self.y_test, y_pred)
        return model_result, y_pred


class CustomTrainer:
    def __init__(self, uploaded_file, typetotrain):
        self.data_class = Data(uploaded_file)
        self.typetotrain = typetotrain
        
    def compute_results(self, y_true, y_pred, com_time):
        if self.typetotrain == 'classification':
            model_res = compute_classification_results(y_pred, y_true, com_time)
        else:
            model_res = compute_regression_results(y_pred, y_true, com_time)
        return model_res
    
    def is_data_clean(self):
        pass
    
    
    def get_data(self, is_clean):
        if not is_clean:
            data = self.data_class.data
        else:
            data = self.data_class.data
        self.x_train, self.x_test, self.y_train, self.y_test, _, _ = self.data_class.dataprocessing(cts=columntransformsplit, custom=True)
    
    
    def trainer(self, model_name, model_parmas=None):
        model = mod_int.model_intializer(model_name, self.typetotrain)()
        if model_parmas is not None:
            model.set_params(**model_parmas)
        model.fit(self.x_train, self.y_train)
        y_pred, test_inf_time = test_predictor(model,self.x_test)
        test_model_result = self.compute_results(self.y_test, y_pred, test_inf_time)
        
        y_pred_train, train_inf_time = train_predictor(model, self.x_train)
        train_model_result = self.compute_results(self.y_train, y_pred_train, train_inf_time)
                    
        model_result = {'train_model_result':train_model_result,
                        'test_model_result':test_model_result,
                        'training_time':train_inf_time
                        }         
        return model_result, model
    
    def optuna_trainer(self, available_params, model_name, parameters, n_trials):
        para = available_params[model_name]
        param_names = [par['Parameter'] for par in para]
        def objective(trial):
                param_dict = {}
                for name, listn in parameters.items():
                    if listn:
                        if name in param_names:
                            param_info = para[param_names.index(name)]
                            dtype = param_info['Dtype']
                            mod_params = []
                            try:
                                if 'str' not in dtype:
                                    if 'int' in dtype:
                                        mod_params = [int(i.strip()) for i in listn]
                                        s = trial.suggest_int(name, mod_params[0], mod_params[1])
                                    elif 'float' in dtype:
                                        mod_params = [float(i.strip()) for i in listn]
                                        s = trial.suggest_float(name, mod_params[0], mod_params[1])
                                    elif 'bool' in dtype:
                                        mod_params = [bool(i.strip()) for i in listn]
                                        s = trial.suggest_categorical(name, mod_params)
                                    
                                else:
                                    mod_params = listn
                                    s = trial.suggest_categorical(name, mod_params)
                            except Exception as e:
                                raise e
                                
                            param_dict[name] = s
                model = mod_int.model_intializer(model_name, self.typetotrain)()
                model.set_params(**param_dict)
                model.fit(self.x_train, self.y_train)

                y_pred = model.predict(self.x_test)
                accuracy = accuracy_score(self.y_test, y_pred)
                
                return accuracy
            
        return params_init.Optuna_tuning(objective, n_trials)
    
    def gridcv_trainer(self,  available_params, model_name, parameters):
        para = available_params[model_name]
        param_names = [par['Parameter'] for par in para]
        param_dict = {}
        for name, listn in parameters.items():
            if listn:
                if name in param_names:
                    param_info = para[param_names.index(name)]
                    dtype = param_info['Dtype']
                    mod_params = []
                    try:
                        if 'str' not in dtype:
                            if 'int' in dtype:
                                mod_params = [int(i.strip()) for i in listn]
                            elif 'float' in dtype:
                                mod_params = [float(i.strip()) for i in listn]
                            elif 'bool' in dtype:
                                mod_params = [bool(i.strip()) for i in listn]
                        else:
                            mod_params = listn
                    except Exception as e:
                        raise e
                        
                    param_dict[name] = mod_params
        
        model = mod_int.model_intializer(model_name, self.typetotrain)()
        search = params_init.Grid_SearchCV(self.typetotrain, model, param_dict)
        search.fit(self.x_train, self.y_train)
        return search
    
    
    def randomcv_trainer(self,  available_params, model_name, parameters):
        para = available_params[model_name]
        param_names = [par['Parameter'] for par in para]
        param_dict = {}
        for name, listn in parameters.items():
            if name in param_names:
                param_info = para[param_names.index(name)]
                dtype = param_info['Dtype']
                mod_params = []
                try:
                    if 'str' not in dtype:
                        if 'int' in dtype:
                            mod_params = [int(i.strip()) for i in listn]
                        elif 'float' in dtype:
                            mod_params = [float(i.strip()) for i in listn]
                        elif 'bool' in dtype:
                            mod_params = [bool(i.strip()) for i in listn]
                    else:
                        mod_params = listn
                except Exception as e:
                    raise e
                    
                param_dict[name] = mod_params
        
        model = mod_int.model_intializer(model_name, self.typetotrain)()
        search = params_init.Randomized_SearchCV(self.typetotrain, model, param_dict)
        search.fit(self.x_train, self.y_train)
        return search
    
    
    def tpot_trainer(self):
        X_train, X_test, y_train, y_test, _, _ = self.data_class.dataprocessing(cts=columntransformsplit, custom=True)
        finder = params_init.tpot_tuner()
        finder.fit(X_train, y_train)
        return finder, finder.score(X_test, y_test)