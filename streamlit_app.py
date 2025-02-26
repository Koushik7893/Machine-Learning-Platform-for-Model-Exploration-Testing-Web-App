import streamlit as st
import numpy as np
import pandas as pd
from src.helper import datasets_dict, json_load, datasets_path
from src.components.DataIngestion import Data
from src.components.ExploreDataset import data_info, data_info2, data_val, charts_page, display_dataset_results, display_model_parameters, display_model_with_parameters, display_stats
from src.components.ExploreModels import datasets_info, display_model_results, display_params, explore_model
from src.components.ExploreCategories import category_datasets_results, category_models_results, category_train, category_clf_params, category_reg_params
from src.pipelines.ModelParams import available_params
from src.pipelines.Models import model_dict


st.set_page_config(layout="wide")
# st.balloons()

def prepare_data_for_display(data, num_c, cat_c):
    data[num_c].fillna(0, inplace=True)
    data[cat_c].fillna('Nan', inplace=True)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in data.columns:
        if data[col].dtype == 'object':  # Convert object columns to string
            data[col] = data[col].astype('string')
        elif data[col].dtype == 'float64':  # Ensure compatibility of float64 columns
            data[col] = data[col].astype('float32')
    return data

    
results = json_load('artifacts/all_results.json')
info = json_load('artifacts/datasets_info.json')
datasets = datasets_path()
    
def dataset(select, dataset_name, types):
    path = datasets_dict()[dataset_name]
    data_class = Data(path)
    data = data_class.data
    num_c, cat_c = data_class.columns_split_for_display()
    
    if select =='explore_data':
        tab1, tab2, tab3 = st.tabs(["Data", "Dataset Info","Statistics Of Data"])
        with tab1:
            data_val(data)
        with tab2:
            data_info(data, num_c, cat_c)
        with tab3:
            display_stats(data)
            
            
    if select =='vizualize_data':
        tab1, tab2 = st.tabs(["Visualize Data", "Dataset Info"])
        with tab1:
            charts_page(prepare_data_for_display(data, num_c, cat_c))
        with tab2:
            data_info2(data, num_c, cat_c)
            
            
    if select =='models_results':
        tab1, tab2, tab3 = st.tabs(["Model Results", "Model Params", "Data Info"])
        with tab1:
            display_dataset_results(results[types][dataset_name])
        with tab2:
            data_info2(data, num_c, cat_c)
        with tab3:
            display_model_with_parameters(available_params, model_dict[types])
      

def model(select, model_name, types):
    params = available_params[model_name]
    if select =='explore_results':
        tab1, tab2, tab3 = st.tabs(["Model Results", "Model Params", "Data Info"])         
        with tab1:
            display_model_results(dic=results[types],model_name=model_name)
        with tab2:
            datasets_info(types, info)
        with tab3:
            display_params(params)
    
    
    if select =='explore_parameters':
        tab1, tab2, tab3 = st.tabs(["Model Results", "Model Params", "Data Info"])
        with tab1:
            display_model_with_parameters(available_params, model_dict[types])
        with tab2:
            datasets_info(types, info)
        with tab3:
            explore_model(types, model_name, params, results[types], model_dict[types])
            
            
    if select =='custom_data':
        tab1, tab2, tab3 = st.tabs(["Model Results", "Model Params", "Data Info"])
        with tab1:
            display_model_results(dic=results[types],model_name=model_name)
        with tab2:
            datasets_info(types, info)
        with tab3:
            explore_model(types, model_name, available_params, model_dict[types][model_name], datasets[types], category_clf_params, category_reg_params)
            

def category(select, category_name, types):
    if select =='explore_results':
        tab1, tab2, tab3 = st.tabs(["Explore Results", "Data Info", "Model Params"])         
        with tab1:
            category_train(available_params, datasets[category_name], model_dict[category_name], category_name)
        with tab2:
            datasets_info(types, info)
        with tab3:
            display_model_with_parameters(available_params, model_dict[types])
    
    
    if select =='explore_models':
        tab1, tab2, tab3 = st.tabs(["Model Results", "Data Info", "Model Params"])
        with tab1:
            category_models_results(results[types], model_dict[category_name])
        with tab2:
            pass
        with tab3:
            display_model_with_parameters(available_params, model_dict[types])
            
            
    if select =='explore_datasets':
        tab1, tab2, tab3 = st.tabs(["Model Results", "Data Info", "Model Params"])
        with tab1:
            category_datasets_results(results[types])
        with tab2:
            datasets_info(types, info)
        with tab3:
            display_model_with_parameters(available_params, model_dict[types])


# Detect page type from query parameters
# Get query parameters safely
dic = st.query_params.to_dict()

page = dic.get('page')  # Get 'page' safely

if page == 'dataset':
    dataset(dic.get('select', ''), dic.get('name', ''), dic.get('types', ''))

elif page == 'model':
    model(dic.get('select', ''), dic.get('name', ''), dic.get('types', ''))

elif page == 'category':
    category(dic.get('select', ''), dic.get('name', ''), dic.get('types', ''))




