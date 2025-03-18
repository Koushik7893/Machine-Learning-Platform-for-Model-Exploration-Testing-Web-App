import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import streamlit as st
from src.components.ExploreDataset import data_val, data_info
from src.helper import safe_float
from src.Trainer import CustomTrainer
import mlflow
import tempfile
import json
import os
import pickle
from io import BytesIO
import optuna
import shap
from src.pipelines.ModelParams import available_params

def display_shap(model, X_test, columns):
    explainer = shap.Explainer(model, X_test)
    shap_values = explainer(X_test)

    st.title("SHAP Explainability Results")
    
    st.subheader("🔍 SHAP Summary Plot - Feature Importance")
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=columns, show=False)
    st.pyplot(fig)
    
    st.subheader("📊 SHAP Bar Plot - Feature Importance")
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, plot_type="bar", feature_names=columns, show=False)
    st.pyplot(fig)

    # if "selected_feature" not in st.session_state:
    #     st.session_state.selected_feature = None

    # st.subheader("🔗 SHAP Dependence Plot - Feature Interaction")

    # # Use session_state to store selected feature
    # feature = st.selectbox("Select Feature", [None] + list(columns), index=0, key="selected_feature")

    # if feature:
    #     fig, ax = plt.subplots(figsize=(10, 6))
    #     shap.dependence_plot(feature, shap_values.values, X_test, feature_names=columns, show=False)
    #     st.pyplot(fig)
    
    
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
  
  
  
def mlflow_track_uri(tracking_uri):
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
                    return MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD
                                
                                
def custom(types, model_name,params, cat_clf_par, cat_reg_par, rearrange_params, display_model_with_parameters):
    uploaded_file = st.file_uploader("Choose a csv file", type=["csv"])
    if uploaded_file is not None:
        CTrainer = CustomTrainer(uploaded_file, types)
        data = CTrainer.data_class.data
        num_c, cat_c = CTrainer.data_class.columns_split_for_display()
        parameters = None
        tracking_uri = None
        is_mlflow = st.toggle("Do you want to log all in MLflow")
        if is_mlflow:
            MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD = mlflow_track_uri(tracking_uri)    
        
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


def optuna_results(study):
    # Assume your study is already created and optimized
    # For example:
    # study = optuna.create_study(direction="maximize")
    # study.optimize(objective, n_trials=20)

    # Title and introductory text
    st.title("Optuna Study Results")
    st.write("A summary of the hyperparameter tuning study using Optuna.")

    # Section: Best Trial Details
    st.header("Best Trial Details")
    st.write("**Best Hyperparameters:**", study.best_params)
    st.write("**Best Objective Value:**", study.best_value)

    best_trial = study.best_trial
    st.write("**Best Trial Number:**", best_trial.number)
    st.write("**Best Trial Value:**", best_trial.value)
    st.write("**Best Trial Parameters:**", best_trial.params)
    st.write("**Best Trial User Attributes:**", best_trial.user_attrs)

    # Section: All Trials Overview
    st.header("All Trials Overview")
    # Convert the trials into a DataFrame and display it
    trials_df = study.trials_dataframe()
    st.dataframe(trials_df)

    st.write("**Total Number of Trials:**", len(study.trials))
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    st.write("**Number of Pruned Trials:**", len(pruned_trials))
    st.write("**Number of Completed Trials:**", len(completed_trials))
    st.write("**Optimization Direction:**", study.direction)

    # Section: Visualizations
    st.header("Visualizations")

    # Plot Optimization History
    st.subheader("Optimization History")
    opt_hist_fig = optuna.visualization.plot_optimization_history(study)
    st.plotly_chart(opt_hist_fig)

    # Plot Parameter Importances
    st.subheader("Parameter Importances")
    param_imp_fig = optuna.visualization.plot_param_importances(study)
    st.plotly_chart(param_imp_fig)

    # Plot Parallel Coordinate
    st.subheader("Parallel Coordinate Plot")
    par_coord_fig = optuna.visualization.plot_parallel_coordinate(study)
    st.plotly_chart(par_coord_fig)



def optuna_params(types, model_name, available_params, display_model_with_parameters):
    uploaded_file = st.file_uploader("Choose a csv file", type=["csv"])
    if uploaded_file is not None:
        CTrainer = CustomTrainer(uploaded_file, types)
        data = CTrainer.data_class.data
        num_c, cat_c = CTrainer.data_class.columns_split_for_display()
        parameters = None
        ntrials = st.text_input('Ener number of trails to run:', value=20)
        
        default_value = json.dumps({par['Parameter']: [] for par in available_params[model_name]}, indent=2)
        help_text = "Please enter valid JSON. Keys and string values must use double quotes. Example:\n" \
            '{  "fit_intercept": ["True", "False"], "C": ["0.01", "10.0"]}'
        
        parameters = st.text_area(
            label="Enter your params: for integers, float give min and max value, for  str you can give how many you need",  
            value=default_value,  
            height=200,  
            max_chars=500,
            help=help_text,  
            placeholder="Check in available params"  
        )
        
        submit = st.button('Ok')
        if submit:
            
            try:
                parameters = json.loads(parameters)
            except json.JSONDecodeError:
                st.error("Invalid JSON format. Please check your input.") 
                
            CTrainer.get_data(True)
            study = CTrainer.optuna_trainer(available_params, model_name, parameters, n_trials=int(ntrials))
            tab1, tab2, tab3 = st.tabs(["Data Info", "Model Results", "Explore Params"])
            with tab1:
                tab_d_1, tab_d_2 = st.tabs(["Data Info", "Data"])
                with tab_d_1:
                    data_val(data)
                with tab_d_2:
                    data_info(data, num_c, cat_c)
            with tab2:
                optuna_results(study)
                
            with tab3:
                display_model_with_parameters(available_params[model_name])
                
def grid_cv_results(grid_search):
    st.header("Grid Search Results")
    st.subheader("Best Parameters & Score")
    st.write("**Best Hyperparameters:**", grid_search.best_params_)
    st.write("**Best Training Score:**", grid_search.best_score_)
    # st.write("**Test Score:**", test_score_grid)

    st.subheader("CV Results Overview")
    cv_results_grid = pd.DataFrame(grid_search.cv_results_)
    st.dataframe(cv_results_grid)

    if 'param_C' in cv_results_grid.columns:
        st.subheader("Mean Test Score vs C")
        fig_grid = px.bar(cv_results_grid, x='param_C', y='mean_test_score', title="Grid Search: Mean Test Score vs C")
        st.plotly_chart(fig_grid)  
        
        
def grid_cv_params(types, model_name, params, display_model_with_parameters, cat_clf_par, cat_reg_par):
    uploaded_file = st.file_uploader("Choose a csv file", type=["csv"])
    if uploaded_file is not None:
        CTrainer = CustomTrainer(uploaded_file, types)
        data = CTrainer.data_class.data
        num_c, cat_c = CTrainer.data_class.columns_split_for_display()
        parameters = None
        
        if types == 'classification':
            parameters = cat_clf_par(params, model_name, custom=True)
        elif types == 'regression':
            parameters = cat_reg_par(params, model_name)
        
        submit = st.button('Ok')
        if submit:
            CTrainer.get_data(True)
            if parameters is not None:
                search = CTrainer.gridcv_trainer(params, model_name, parameters)
                tab1, tab2, tab3 = st.tabs(["Data Info", "Model Results", "Explore Params"])
                with tab1:
                    tab_d_1, tab_d_2 = st.tabs(["Data Info", "Data"])
                    with tab_d_1:
                        data_val(data)
                    with tab_d_2:
                        data_info(data, num_c, cat_c)
                with tab2:
                    grid_cv_results(search)
                    display_shap(search.best_estimator_.predict,CTrainer.x_test, CTrainer.data_class.get_x_columns())
                with tab3:
                    display_model_with_parameters(params[model_name])
                
def random_cv_results(random_search):
    st.header("Random Search Results")
    st.subheader("Best Parameters & Score")
    st.write("**Best Hyperparameters:**", random_search.best_params_)
    st.write("**Best Training Score:**", random_search.best_score_)
    # st.write("**Test Score:**", test_score_random)

    st.subheader("CV Results Overview")
    cv_results_rand = pd.DataFrame(random_search.cv_results_)
    st.dataframe(cv_results_rand)

    if 'param_C' in cv_results_rand.columns:
        st.subheader("Mean Test Score vs C")
        fig_rand = px.bar(cv_results_rand, x='param_C', y='mean_test_score', title="Random Search: Mean Test Score vs C")
        st.plotly_chart(fig_rand)
         
def random_cv_params(types, model_name, params, display_model_with_parameters, cat_clf_par, cat_reg_par):
    uploaded_file = st.file_uploader("Choose a csv file", type=["csv"])
    if uploaded_file is not None:
        CTrainer = CustomTrainer(uploaded_file, types)
        data = CTrainer.data_class.data
        num_c, cat_c = CTrainer.data_class.columns_split_for_display()
        parameters = None
        
        if types == 'classification':
            parameters = cat_clf_par(params, model_name, custom=True)
        elif types == 'regression':
            parameters = cat_reg_par(params, model_name)
        
        submit = st.button('Ok')
        if submit:
            CTrainer.get_data(True)
            if parameters is not None:
                search = CTrainer.gridcv_trainer(params, model_name, parameters)
                tab1, tab2, tab3 = st.tabs(["Data Info", "Model Results", "Explore Params"])
                with tab1:
                    tab_d_1, tab_d_2 = st.tabs(["Data Info", "Data"])
                    with tab_d_1:
                        data_val(data)
                    with tab_d_2:
                        data_info(data, num_c, cat_c)
                with tab2:
                    random_cv_results(search)
                    display_shap(search.best_estimator_.predict,CTrainer.x_test, CTrainer.data_class.get_x_columns())
                    
                with tab3:
                    display_model_with_parameters(params[model_name])


def tpot_results(tpot, test_score_tpot):
    st.title("TPOT AutoML Results")

    st.subheader("Best Pipeline & Test Score")
    st.write("**Test Score:**", test_score_tpot)

    st.subheader("Export Best Pipeline")
    # Save the TPOT pipeline as a Python script
    pipeline_code = tpot.export()
    st.code(pipeline_code, language="python")

    # Save pipeline as a pickle file for download
    pickle_buffer = BytesIO()
    pickle.dump(tpot.fitted_pipeline_, pickle_buffer)
    pickle_buffer.seek(0)

    st.download_button(
        label="Download Best Model (Pickle)",
        data=pickle_buffer,
        file_name="best_tpot_model.pkl",
        mime="application/octet-stream"
    )  
    
    
    
def tpot_searcher(types, model_name, params, display_model_with_parameters, model_dict):
    uploaded_file = st.file_uploader("Choose a csv file", type=["csv"])
    if uploaded_file is not None:
        CTrainer = CustomTrainer(uploaded_file, types)
        data = CTrainer.data_class.data
        num_c, cat_c = CTrainer.data_class.columns_split_for_display()

        tpot, test_score_tpot = CTrainer.tpot_trainer()

        tab1, tab2, tab3 = st.tabs(["Data Info", "Model Results", "Explore Params"])
        with tab1:
            tab_d_1, tab_d_2 = st.tabs(["Data Info", "Data"])
            with tab_d_1:
                data_val(data)
            with tab_d_2:
                data_info(data, num_c, cat_c)
        with tab2:
            tpot_results(tpot, test_score_tpot)
            
        with tab3:
            display_model_with_parameters(params, model_dict)
                          
              
