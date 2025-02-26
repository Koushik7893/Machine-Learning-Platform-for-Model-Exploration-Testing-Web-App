import os
import pickle
import json
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


def datasets_path():
    data = {}
    types = os.listdir(path='data')
    for typeofclass in types:
        paths = {}
        datasets = os.listdir(path=f'data/{typeofclass}')
        for dataset in datasets:
            if dataset.endswith('.csv'):
                paths[str.title(os.path.splitext(dataset)[0])] = os.path.join(os.getcwd(),f'data/{typeofclass}/{dataset}')
        data[typeofclass] = paths
    return data

def datasets_dict():
    all_datasets = datasets_path()
    paths = {name: path for _,dicta  in all_datasets.items() for name, path in dicta.items()}
    return paths


def datasets_info():
    import pandas as pd
    all_info = {}
    dic = datasets_path()
    for types, datasets in dic.items():
        if types not in all_info.keys():
            all_info[types] = {}
        for dataset, paths in datasets.items():
            if dataset not in all_info[types].keys():
                data = pd.read_csv(paths)
                if 'Unnamed: 0' in data.columns:
                    data.drop(columns='Unnamed: 0', inplace=True)
                data_dic = {'Shape':data.shape, 'No.of Features':data.shape[1]-1, 'Size of Data':data.shape[0], 'Features':list(data.columns)}
                all_info[types][dataset] = data_dic
    os.makedirs(os.path.dirname('artifacts/datasets_info.json'), exist_ok=True)
    with open('artifacts/datasets_info.json', "w") as file:
        json.dump(all_info, file)


def save_pkl_file(file_path, model):
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)
    with open(file_path, "wb") as file:
        pickle.dump(model,file)
        
        
def load_pkl_file(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)
    
def compute_classification_results(y_true, y_pred):
    results = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'), 
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted'),
    #   'classification_report': classification_report(y_true, y_pred, output_dict=True),  
        'roc_auc': roc_auc_score(y_true, y_pred, average='weighted', multi_class='ovr') if len(set(y_true)) == 2 else 'N/A'
    }
    return results

def compute_regression_results(y_true, y_pred):
    results = {
        'mae': mean_absolute_error(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': root_mean_squared_error(y_true, y_pred),  
        'r2_score': r2_score(y_true, y_pred),
    }
    return results

def json_as(path, results):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as file:
        json.dump(results, file)
        
def json_load(path):
    with open(path, "r") as file:
        data = json.load(file)
    return data

def all_results_json(results):
    path = 'artifacts/all_results.json'
    if os.path.exists(path):
        loaded_results = json_load(path)
        for types, datasets in results.items():
            if types not in loaded_results.keys():
                loaded_results[types] = {}
            for dataset, models in datasets.items():
                if dataset not in loaded_results[types].keys():
                    loaded_results[types][dataset] = {}
                for model, model_results in models.items():
                    if model not in loaded_results[types][dataset].keys():
                        loaded_results[types][dataset][model] = model_results
                    else:
                        loaded_results[types][dataset][model] = model_results
        return json_as(path=path, results=loaded_results)
                
    else:
        return json_as(path=path, results=results)