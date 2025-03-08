import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from src.helper import safe_float



def Scatter_Plot(data, x, y, hue=None, legend='brief'):
    fig, ax = plt.subplots(figsize=(7,5))
    sns.scatterplot(data=data, x=x, y=y, hue=hue, legend=legend, ax=ax)
    return fig

def Line_Plot(data, x, y, hue=None, legend='brief'):
    fig, ax = plt.subplots(figsize=(7,5))
    sns.lineplot(data=data, x=x, y=y, hue=hue, legend=legend, ax=ax)
    return fig

def Bar_Plot(data, x, y, hue=None, order=None, legend=None):
    fig, ax = plt.subplots(figsize=(7,5))
    if order is None:
        sns.barplot(data=data, x=x, y=y, hue=hue, legend=legend, ax=ax)
    else:
        order=data.sort_values(y,ascending = False).x
        sns.barplot(data=data, x=x, y=y, order=order, hue=hue, legend=legend, ax=ax)
    return fig

def Count_Plot(data, x, hue=None):
    fig, ax = plt.subplots(figsize=(7,5))
    sns.countplot(data=data, x=x, hue=hue, ax=ax)
    return fig

def Box_Plot(data, x, y, hue=None, orient=None):
    fig, ax = plt.subplots(figsize=(7,5))
    sns.boxplot(data=data, x=x, y=y, hue=hue, orient=orient, ax=ax)
    return fig
 
def Histogram_Plot(data,col, hue=None):
    fig, ax = plt.subplots(figsize=(7,5))
    sns.histplot(data[col],hue=hue, ax=ax)
    return fig
   
def Pair_Plot(data, hue=None, selected_vars=None):
    if selected_vars or hue:
        return sns.pairplot(data=data, hue=hue, vars=selected_vars)
    return sns.pairplot(data=data)

def Correlation_Map(data):
    fig, ax = plt.subplots(figsize=(7,5))
    cor_matrix = data.corr(numeric_only=True)
    sns.heatmap(cor_matrix, cmap='GnBu_r', annot=True, ax=ax)
    return fig

def data_info(data, num_c, cat_c):
    co = st.columns([0.1,0.9])
    co[0].button('Back')
    if co[0].button('Full Info'):
        for col in data.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            if col in num_c:
                sns.histplot(data[col], ax=ax, kde=True)  
            else:  
                sns.histplot(data = data, y = data[col], ax=ax)
            ax.set_xlabel('')
            ax.set_ylabel('')
            with co[1]:
                _, graph, _, inf = st.columns([0.1,0.37,0.05,0.37])
                with graph:
                    st.pyplot(fig)
                with inf:
                    na_val = data[col].isna().sum()
                    valid_count = len(data[col]) - na_val
                    invalid_count = na_val
                    
                    if col in cat_c:
                        unique_count = data[col].nunique()
                        mc = data[col].value_counts()
                        mst_com = ', '.join([f"{a} ({b})" for a, b in list(mc.items())[:3]])
                        ind = ['Name', 'Valid', 'Invalid', 'Unique', 'Most Common', 'Dtype']
                        val = [col.title(), valid_count, invalid_count, unique_count, mst_com, 'object']
                        
                    else:
                        max_val = data[col].max()
                        min_val = data[col].min()
                        Mean = data[col].mean()
                        Std = data[col].std()
                        ind = ['Name', 'Valid', 'Invalid', 'Min - Max Value', 'Mean', 'Std', 'Dtype']
                        val = [col.title(), valid_count, invalid_count, f'{min_val}, {max_val}', f'{Mean:.2f}', f'{Std:.2f}', data[col].dtype]
                    summary_df = pd.DataFrame({'Metric': ind, 'Value': val})
                    st.markdown("""<style>.dataframe-container {width: 90%; height: 200px;: auto;}</style>""", unsafe_allow_html=True)
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)      
    else:
        with co[1]:
            data_info2(data, num_c, cat_c)
            
            
def data_info2(data, num_c, cat_c):
    data[num_c] = data[num_c].fillna(0)
    data[cat_c] = data[cat_c].fillna('Nan')
    cat_df = pd.DataFrame({'columns':[col.title() for col in cat_c], 
                       'dtype':[data[col].dtype for col in cat_c],
                       'frequency':[list(data[col].value_counts().items())[0][0] for col in cat_c],
                       'values':[dict(data[col].value_counts().items()).values() for col in cat_c]
                       })
    num_df = pd.DataFrame({'columns':[col.title() for col in num_c],
                           'dtype':[data[col].dtype for col in num_c],
                           'min_value':[data[col].min() for col in num_c],
                           'max_value':[data[col].max() for col in num_c],
                           'values':[list(data[col]) for col in num_c]
                           })
    st.subheader('Numerical Columns', divider=True)
    st.dataframe(cat_df, column_config={'columns':'Column Names',
                                        'dtype':'Dtype',
                                        'frequency':'Most Frequent',
                                        'values':st.column_config.BarChartColumn(
                                            'All Values',
                                            )
                                        },hide_index=True, use_container_width=True
                 )
    st.subheader('Categorical Columns', divider=True)
    st.dataframe(num_df, column_config={'columns':'Column Names',
                                        'dtype':'Dtype',
                                        'min_value':'Min Value',
                                        'max_value':'Max Value',
                                        'values':st.column_config.LineChartColumn(
                                            'All Values',width='large'
                                            )
                                        },hide_index=True, use_container_width=True
                 )

def data_val(data):
    col = st.columns([0.9,0.1])
    
    col[1].button('Done', type='secondary', use_container_width=True)
    if col[1].button('Edit', type='secondary', use_container_width=True):
        col[0].data_editor(data, use_container_width=True)
    else:
        col[0].dataframe(data, use_container_width=True)

def charts_page(data):  
    charts_list = ['Scatter Plot', 'Line Plot', 'Bar Plot', 'Count Plot', 'Box Plot', 'Histogram Plot', 'Pair Plot', 'Correlation Map']
    chart_options = st.selectbox('Choose a chart to display:',  [None] + charts_list, placeholder='Select Chart')
    if chart_options:
        if chart_options == 'Scatter Plot':
            c1, c2, c3 = st.columns([0.3,0.2,0.5])
            x = c1.selectbox('Select X-axis variable:', [None] + list(data.columns), placeholder='Choose X-axis')
            y = c1.selectbox('Select Y-axis variable:', [None] + list(data.columns), placeholder='Choose Y-axis')
            hue = c1.selectbox('Color by variable (optional):', [None] + list(data.columns), placeholder='Choose category')
            if x is not None and y is not None:
                c3.pyplot(Scatter_Plot(data, x, y, hue))
        
        elif chart_options == 'Line Plot':
            c1, c2, c3 = st.columns([0.3,0.2,0.5])
            x = c1.selectbox('Select X-axis variable:', [None] + list(data.columns), placeholder='Choose X-axis')
            y = c1.selectbox('Select Y-axis variable:', [None] + list(data.columns), placeholder='Choose Y-axis')
            hue = c1.selectbox('Color by variable (optional):', [None] + list(data.columns), placeholder='Choose category')
            if x and y:
                c3.pyplot(Line_Plot(data, x, y, hue))
        
        elif chart_options == 'Bar Plot':
            c1, c2, c3 = st.columns([0.3,0.2,0.5])
            x = c1.selectbox('Select X-axis variable:', [None] + list(data.columns), placeholder='Choose X-axis')
            y = c1.selectbox('Select Y-axis variable:', [None] + list(data.columns), placeholder='Choose Y-axis')
            hue = c1.selectbox('Color by variable (optional):', [None] + list(data.columns), placeholder='Choose category')
            if x and y:
                c3.pyplot(Bar_Plot(data, x, y, hue))
        
        elif chart_options == 'Count Plot':
            c1, c2, c3 = st.columns([0.3,0.2,0.5])
            x = c1.selectbox('Select X-axis variable:', [None] + list(data.columns), placeholder='Choose X-axis')
            hue = c1.selectbox('Color by variable (optional):', [None] + list(data.columns), placeholder='Choose category')
            if x:
                c3.pyplot(Count_Plot(data, x, hue))
        
        elif chart_options == 'Box Plot':
            c1, c2, c3 = st.columns([0.3,0.2,0.5])
            x = c1.selectbox('Select X-axis variable:', [None] + list(data.columns), placeholder='Choose X-axis')
            y = c1.selectbox('Select Y-axis variable:', [None] + list(data.columns), placeholder='Choose Y-axis')
            hue = c1.selectbox('Color by variable (optional):', [None] + list(data.columns), placeholder='Choose category')
            orient = c1.selectbox('Orientation:', [None] + ['Vertical', 'Horizontal'], placeholder='Choose orientation')
            if x is not None and y is not None:
                c3.pyplot(Box_Plot(data, x, y, hue, orient))
        
        elif chart_options == 'Histogram Plot':
            c1, c2, c3 = st.columns([0.3,0.2,0.5])
            col = c1.selectbox('Select variable for histogram:', [None] + list(data.columns), placeholder='Choose variable')
            hue = c1.selectbox('Color by variable (optional):', [None] + list(data.columns), placeholder='Choose category')
            if col is not None:
                c3.pyplot(Histogram_Plot(data, col, hue))
        
        elif chart_options == 'Pair Plot':
            c1, c2 = st.columns(2)
            hue = c1.selectbox('Color by variable (optional):', [None] + list(data.columns), placeholder='Choose category')
            selected_vars = c2.multiselect('Select variables to plot:', [None] + list(data.columns), placeholder='Choose variables')
            if selected_vars and len(data.columns) > 10:
                st.exception('Due to increase in number of columns as execution time increases please select some variables')
                st.pyplot(Pair_Plot(data, hue, selected_vars))
            else:
                st.pyplot(Pair_Plot(data),hue, selected_vars)
        
        elif chart_options == 'Correlation Map':
            st.pyplot(Correlation_Map(data)) 
            
def display_dataset_results(dic):
    data_ref = []
    for model, model_results in dic.items():
        train_results = model_results["train_model_result"]
        test_results = model_results["test_model_result"]
        
        row = {
            "Model": model,
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

        
    
        data_ref.append(row)
    st.dataframe(pd.DataFrame(data_ref))
    
def display_stats(data):
    st.dataframe(pd.DataFrame(data.describe()).T, use_container_width=True)
        
def display_model_parameters(params_dict, model_dict):
    model_ind = []
    param_name_ind = []
    value_series = []
    for model, params_list in params_dict.items():
        if model in model_dict.keys():
            for params in params_list:
                for param, value in params.items():
                    model_ind.append(model)
                    param_name_ind.append(param)
                    value_series.append(value)
    model_params_arrays = [model_ind, param_name_ind]
    tuples = list(zip(*model_params_arrays))
    index = pd.MultiIndex.from_tuples(tuples, names=['Model Name', 'Parameter Name'])
    st.dataframe(pd.DataFrame(value_series, index=index))

def display_model_with_parameters(params_dict, model_dict):
    col = st.columns([0.4,0.7])
    st.markdown("""
        <style>
        .custom-button {height: 40px;width: 100px; font-weight: bold; color: white; background-color: #4CAF50; border: none; border-radius: 8px; text-align: center;}
        .custom-dataframe {height: 100px;  /* Set consistent height for all dataframes */}
        </style>""", unsafe_allow_html=True)

    # Loop through each model to create buttons with uniform style
    for model_name in model_dict.keys():
        if col[0].button(model_name, key=model_name, help=f"Show parameters for {model_name}"):
            # Show the dataframe with a fixed height
            col[1].markdown(
                f"<div class='custom-dataframe'>{pd.DataFrame(params_dict[model_name]).to_html(index=False)}</div>",
                unsafe_allow_html=True)