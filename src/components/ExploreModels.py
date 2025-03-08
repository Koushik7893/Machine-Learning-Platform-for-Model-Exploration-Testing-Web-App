import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from src.components.ExploreDataset import data_val, data_info
from src.helper import safe_float
from src.Trainer import CustomTrainer
import mlflow
import tempfile
import json
import os
from src.pipelines.ModelParams import available_params

    
def display_model_results(dic, model_name):
    data = []
    for dataset, models in dic.items():
        if model_name in models:  
            model_results = models[model_name]
            train_results = model_results["train_model_result"]
            test_results = model_results["test_model_result"]

            row = {
                "Dataset": dataset,
                "Train Accuracy": round(train_results["accuracy"], 4),
                "Train Precision": round(train_results["precision"], 4),
                "Train Recall": round(train_results["recall"], 4),
                "Train F1-Score": round(train_results["f1_score"], 4),
                "Train ROC AUC": round(safe_float(train_results.get("roc_auc", np.nan)), 4),
                "Train Inference Time": round(train_results["inf_time"], 6),  # Kept more precision for small values
                "Test Accuracy": round(test_results["accuracy"], 4),
                "Test Precision": round(test_results["precision"], 4),
                "Test Recall": round(test_results["recall"], 4),
                "Test F1-Score": round(test_results["f1_score"], 4),
                "Test ROC AUC": round(safe_float(test_results.get("roc_auc", np.nan)), 4),
                "Test Inference Time": round(test_results["inf_time"], 6),
                "Training Time": round(model_results["training_time"], 4),
                "File Size (MB)": round(model_results["file_size_mb"] / 1024, 4),
            }

                
            data.append(row)
    st.dataframe(pd.DataFrame(data))
    
def datasets_info(types, info):
    data = pd.DataFrame(info[types]).T
    st.dataframe(data.sort_values(['Size of Data'], ascending=True))
    
def display_params(params):
    dic = pd.DataFrame(params)
    st.dataframe(dic, hide_index=True)

def explore_model(types, model_name, params, model, datasets, clf_par, reg_par):
    if model_name in  model.keys():
        if types == 'classification':
            parameters = clf_par(params, model_name)
        elif types == 'regression':
            parameters = reg_par(params, model_name)
    if parameters is not None:
        if model_name == 'LinearRegression':
            st.header("Linear Regression")

def get_row(model_results,train_res):
    test_results = model_results["test_model_result"]
    if train_res:
        train_results = model_results["train_model_result"]
    
        row = {
                "Train Accuracy": round(train_results["accuracy"], 4),
                "Train Precision": round(train_results["precision"], 4),
                "Train Recall": round(train_results["recall"], 4),
                "Train F1-Score": round(train_results["f1_score"], 4),
                "Train ROC AUC": round(safe_float(train_results.get("roc_auc", np.nan)), 4),
                "Train Inference Time": round(train_results["inf_time"], 6),  # Kept more precision for small values
                "Test Accuracy": round(test_results["accuracy"], 4),
                "Test Precision": round(test_results["precision"], 4),
                "Test Recall": round(test_results["recall"], 4),
                "Test F1-Score": round(test_results["f1_score"], 4),
                "Test ROC AUC": round(safe_float(test_results.get("roc_auc", np.nan)), 4),
                "Test Inference Time": round(test_results["inf_time"], 6),
                "Training Time": round(model_results["training_time"], 4),
                }
    else:
        row = {
                "Test Accuracy": round(test_results["accuracy"], 4),
                "Test Precision": round(test_results["precision"], 4),
                "Test Recall": round(test_results["recall"], 4),
                "Test F1-Score": round(test_results["f1_score"], 4),
                "Test ROC AUC": round(safe_float(test_results.get("roc_auc", np.nan)), 4),
                "Test Inference Time": round(test_results["inf_time"], 6),
                "Training Time": round(model_results["training_time"], 4),
                }
    return row
   
def display_model_result(model_results, model_name, parameter_there, train_res=False):
    data = []
    names = []
    if parameter_there:
        for param_name, dic in model_results.items():
            row = get_row(dic['Results'],train_res)
            data.append(row)
            names.append(param_name)
        st.dataframe(pd.DataFrame(data, index=names))
    else:
        row = get_row(model_results,train_res)
        st.dataframe(pd.DataFrame([row], index=[model_name]).T)
  
def custom(types, model_name,params, cat_clf_par, cat_reg_par, rearrange_params,display_model_with_parameters):
    uploaded_file = st.file_uploader("Choose a csv file", type=["csv"])
    if uploaded_file is not None:
        CTrainer = CustomTrainer(uploaded_file, types)
        data = CTrainer.data_class.data
        num_c, cat_c = CTrainer.data_class.columns_split_for_display()
        parameters = None
        tracking_uri = None
        is_mlflow = st.toggle("Do you want to log all in MLflow")
        if is_mlflow:
            tracking_uri = st.text_input('MLFlow Tracking Uri', value='https://dagshub.com/<username>/<repo_name>.mlflow')
            if tracking_uri is not None:
                MLFLOW_TRACKING_USERNAME = st.text_input('DagsHub Username')
                MLFLOW_TRACKING_PASSWORD = st.text_input('MLFlow Tracking Password')
                tracking_exp = st.text_input('MLFlow Experiment')
                
                if st.button('Set'):
                    if MLFLOW_TRACKING_USERNAME is not None:
                        if MLFLOW_TRACKING_PASSWORD is not None:
                            os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_TRACKING_USERNAME
                            os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_TRACKING_PASSWORD
                            mlflow.set_tracking_uri(tracking_uri)
                            if tracking_exp is not None:
                                st.write('set_exp'+ tracking_exp)
                                mlflow.set_experiment(tracking_exp)
                
                
        is_clean = st.toggle("Is given Dataset is Cleaned or not if cleaned turn on it")
        get_train_results = st.toggle("Get Train Data Results Too")
        model_parmas = st.toggle("If you want model params")
        if model_parmas:
            if types == 'classification':
                parameters = cat_clf_par(params, model_name, custom=True)
            elif types == 'regression':
                parameters = cat_reg_par(params, model_name)
        submit = st.button('Ok')
        
        if submit:
            CTrainer.get_data(is_clean)
            if parameters is not None:
                model_result={}
                all_combinations, num_combinations = rearrange_params(params, model_name, parameters)
                rows = (num_combinations + 1) // 2
                if tracking_uri is not None:
                    if MLFLOW_TRACKING_USERNAME is not None: 
                        if MLFLOW_TRACKING_PASSWORD is not None:
                            with mlflow.start_run():
                                for i in range(rows):
                                    for j in range(2):
                                        idx = i * 2 + j
                                        if idx < num_combinations:
                                            params_st = all_combinations[idx]
                                            params_name = " | ".join([f"{key.capitalize()}: {value}" for key, value in params_st.items()])
                                            results, model = CTrainer.trainer(model_name, params_st)
                                            with mlflow.start_run(nested=True):
                                                mlflow.set_tag(f"Parameters of {model_name}", params_name)
                                                mlflow.log_params(params_st)
                                                mlflow.sklearn.log_model(model, f"{model_name} - Model  --- {params_name}")
                                                row = get_row(results, True)
                                                
                                                for key, value in row.items():
                                                    mlflow.log_metric(key, value)
                                                    
                                            param_result = {"Params":params_st, "Results":results}
                                            model_result[params_name] = param_result
                else:
                    for i in range(rows):
                        for j in range(2):
                            idx = i * 2 + j
                            if idx < num_combinations:
                                params_st = all_combinations[idx]
                                params_name = " | ".join([f"{key.capitalize()}: {value}" for key, value in params_st.items()])
                                results, model = CTrainer.trainer(model_name, params_st)
                                param_result = {"Params":params_st, "Results":results}
                                model_result[params_name] = param_result
            else:
                if tracking_uri is not None:
                    if MLFLOW_TRACKING_USERNAME is not None: 
                        if MLFLOW_TRACKING_PASSWORD is not None:
                            with mlflow.start_run():
                                model_result, model = CTrainer.trainer(model_name)
                                mlflow.sklearn.log_model(model, "model")
                                row = get_row(model_result, True)
                                for key, value in row.items():
                                    mlflow.log_metric(key, value)
                else:
                    model_result, _ = CTrainer.trainer(model_name)
                    
            if tracking_uri is not None:
                if MLFLOW_TRACKING_USERNAME is not None: 
                    if MLFLOW_TRACKING_PASSWORD is not None:        
                        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp_file:
                            json.dump(model_result, tmp_file)
                            tmp_filename = tmp_file.name
                        mlflow.log_artifact(tmp_filename)
                        os.remove(tmp_filename)
                
            tab1, tab2, tab3 = st.tabs(["Data Info", "Model Results", "Explore Params"])
            with tab1:
                tab_d_1, tab_d_2 = st.tabs(["Data Info", "Data"])
                with tab_d_1:
                    data_val(data)
                with tab_d_2:
                    data_info(data, num_c, cat_c)
            with tab2:
                parameter_there = True if parameters is not None else False
                if not get_train_results:
                    display_model_result(model_result, model_name, parameter_there)
                else:
                    display_model_result(model_result, model_name, parameter_there, train_res=True)
                
            with tab3:
                display_model_with_parameters(available_params[model_name])

