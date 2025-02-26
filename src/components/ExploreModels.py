import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from src.Trainer import CategoryTrainer

    
def display_model_results(dic, model_name):
    res = {}
    for dataset_name, models in dic.items():
        res[dataset_name] = models[model_name]
    st.dataframe(pd.DataFrame(res).T)
    
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
            


    

