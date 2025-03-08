import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import streamlit as st
import seaborn as sns
from itertools import product
from src.helper import safe_float
from src.Trainer import CategoryTrainer

def plot_fitted_vs_original(y_true, y_pred):
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, color='blue', label='Fitted')
    ax.plot(y_true, y_true, color='red', label='Original', linewidth=2)
    ax.set_xlabel('Original Data')
    ax.set_ylabel('Fitted Data')
    ax.legend()
    st.pyplot(fig)
    
def classification_cov_matrix(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, ax=ax, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted Classes')
    plt.ylabel('Actual Classes')
    st.pyplot(fig)

    
def category_datasets_results(dic):
    for dataset_name, models in dic.items():
        st.header(dataset_name)
        
        col1, col2 = st.columns(2)
        
        data = []
        for model_name, model_results in models.items():
            train_results = model_results.get("train_model_result", {})
            test_results = model_results.get("test_model_result", {})

            data.append({
                "Model": model_name,
                "Train Accuracy": round(train_results.get("accuracy", 0), 4),
                "Test Accuracy": round(test_results.get("accuracy", 0), 4),
                "Train Precision": round(train_results.get("precision", 0), 4),
                "Test Precision": round(test_results.get("precision", 0), 4),
                "Train Recall": round(train_results.get("recall", 0), 4),
                "Test Recall": round(test_results.get("recall", 0), 4),
                "Train F1-Score": round(train_results.get("f1_score", 0), 4),
                "Test F1-Score": round(test_results.get("f1_score", 0), 4),
                "Train ROC AUC": round(safe_float(train_results.get("roc_auc", np.nan)), 4),
                "Test ROC AUC": round(safe_float(test_results.get("roc_auc", np.nan)), 4),
                "Train Inference Time": round(train_results.get("inf_time", 0), 6),  # Higher precision
                "Test Inference Time": round(test_results.get("inf_time", 0), 6),
                "Training Time": round(model_results.get("training_time", 0), 4),
                "File Size (MB)": round(model_results.get("file_size_mb", 0) / 1024, 4),
            })
        
        metrics_df = pd.DataFrame(data).set_index("Model")
        
        col1.dataframe(metrics_df)
        
        fig, ax = plt.subplots()
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black") 
        
        ax.barh(metrics_df.index[::-1], metrics_df["Test Accuracy"][::-1], color="#045e54") 
        ax.set_xlabel("Accuracy", color="white")
        ax.tick_params(axis='x', colors="white")  
        ax.tick_params(axis='y', colors="white") 
        col2.pyplot(fig)

        
def category_models_results(results, models):
    for model in models.keys():
        results_mod = {}

        for dataset_name, dataset_models in results.items():
            if model in dataset_models:  # Check if model exists in dataset
                train_results = dataset_models[model].get("train_model_result", {})
                test_results = dataset_models[model].get("test_model_result", {})

                results_mod[dataset_name] = {
                    "Train Accuracy": round(train_results.get("accuracy", 0), 4),
                    "Test Accuracy": round(test_results.get("accuracy", 0), 4),
                    "Train Precision": round(train_results.get("precision", 0), 4),
                    "Test Precision": round(test_results.get("precision", 0), 4),
                    "Train Recall": round(train_results.get("recall", 0), 4),
                    "Test Recall": round(test_results.get("recall", 0), 4),
                    "Train F1-Score": round(train_results.get("f1_score", 0), 4),
                    "Test F1-Score": round(test_results.get("f1_score", 0), 4),
                    "Train ROC AUC": round(safe_float(train_results.get("roc_auc", np.nan)), 4),
                    "Test ROC AUC": round(safe_float(test_results.get("roc_auc", np.nan)), 4),
                    "Train Inference Time": round(train_results.get("inf_time", 0), 6),  # More precision for small times
                    "Test Inference Time": round(test_results.get("inf_time", 0), 6),    # More precision for small times
                    "Training Time": round(dataset_models[model].get("training_time", 0), 4),
                    "File Size (MB)": round(dataset_models[model].get("file_size_mb", 0) / 1024, 4),
                }

        if not results_mod:
            continue

        st.header(model)
        col1, col2 = st.columns(2)

        metrics_df = pd.DataFrame(results_mod).T
        col1.dataframe(metrics_df)

        accuracy_df = metrics_df[["Test Accuracy"]].sort_values(by="Test Accuracy", ascending=False)
        col2.bar_chart(accuracy_df)


def category_clf_params(params, models_options, custom=False):
    param = params[models_options]
    c1, c2, c3, c4 = st.columns(4)
    if len([par['Parameter'] for par in param]) > 4:
        c5, c6, c7, c8 = st.columns(4)  
        
    if models_options == 'LogisticRegression':
        penalty = c1.multiselect('Select type of penalty', options=[None, 'l1', 'l2', 'elasticnet'], placeholder='Default: l2', help="Specify the norm of the penalty: None, 'l1', 'l2', or 'elasticnet'. Default is 'l2'.")
        dual = c2.multiselect('Dual or primal formulation', options=['True', 'False'], placeholder='Default: False', help="Dual formulation is only implemented for l2 penalty with liblinear solver. Default: False.")
        C = c3.text_input('Inverse of regularization strength', value='1.0', help="Must be a positive float. Smaller values specify stronger regularization. Default: 1.0.").split(',')
        fit_intercept = c4.multiselect('Calculate intercept', options=['True', 'False'], placeholder='Default: True', help="Specifies if a constant (a.k.a. bias or intercept) should be added. Default: True.")
        solver = c6.multiselect('Algorithm for optimization', options=['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'], placeholder='Default: lbfgs', help="Algorithm to use in the optimization problem. Default is 'lbfgs'.")
        max_iter = c5.text_input('Maximum number of iterations', value='100', help="Maximum number of iterations for solvers to converge. Default: 100.").split(',')
        n_jobs = c7.text_input('Number of parallel jobs', value='-1', help="Number of CPU cores used when parallelizing. None means 1, -1 means using all processors.").split(',')
        parameters = {'penalty':penalty, 'dual':dual, 'C':C, 'fit_intercept':fit_intercept, 'solver':solver, 'max_iter':max_iter, 'n_jobs':n_jobs}

    elif models_options == 'SVC':
        C = c1.text_input('Regularization parameter', '1.0', help="Regularization parameter, default=1.0").split(',')
        kernel = c2.multiselect('Kernel type', ['linear', 'poly', 'rbf', 'sigmoid'], default='rbf', help="Kernel type (linear, poly, rbf, sigmoid). Default is 'rbf'.")
        degree = c3.text_input('Degree of polynomial kernel', '3', help="Degree for polynomial kernel. Default is 3.").split(',')
        gamma = c4.multiselect('Kernel coefficient', ['scale', 'auto'], default='scale', help="Kernel coefficient, default is 'scale'.")
        probability = c5.multiselect('Enable probability estimates',['True', 'False'],  default='False', help="Enables probability estimates. Default is False.")
        parameters = {'kernel':kernel, 'degree':degree, 'C':C, 'gamma':gamma, 'probability':probability}

    elif models_options=='GaussianNB':
        var_smoothing = c1.text_input('Variance smoothing', '1e-9', help="Portion of the largest variance of all features added to variances. Default is 1e-9.").split(',')
        parameters = {'var_smoothing':var_smoothing}

    elif models_options=='KNeighborsClassifier':
        n_neighbors = c1.text_input('Number of neighbors', '5', help="Enter a comma-separated list of numbers for the number of neighbors. Default is 5.").split(',')
        weights=c2.multiselect('Weight function',options=['uniform','distance'],default = 'uniform', help="Weight function used in prediction. Default is 'uniform'.")
        algorithm=c3.multiselect('Algorithm for nearest neighbors',options=['auto','ball_tree','kd_tree','brute'],default = 'auto',help="Algorithm used to compute the nearest neighbors. Default is 'auto'.")
        leaf_size = c4.text_input('Leaf size', '30', help="Enter a comma-separated list of numbers for the leaf size. Default is 30.").split(',')
        parameters = {'n_neighbors':n_neighbors, 'weights':weights, 'algorithm':algorithm, 'leaf_size':leaf_size}

    elif models_options=='DecisionTreeClassifier':
        criterion = c1.multiselect('Criterion for splitting', options=['gini', 'entropy'], default='gini', help="Function to measure split quality ('gini', 'entropy').")
        splitter = c2.multiselect('Splitter for tree', options=['best', 'random'], default='best', help="Strategy to split nodes ('best', 'random').")
        max_depth = c3.text_input('Maximum depth of the tree', help="Maximum depth of the tree. Default is None.").split(',')
        min_samples_split = c4.text_input('Minimum samples required to split a node', min_value=2, value=2, step=1, help="Minimum number of samples required to split an internal node.").split(',')
        parameters = {'criterion':criterion, 'splitter':splitter, 'max_depth':max_depth, 'min_samples_split':min_samples_split}

    elif models_options=='RandomForestClassifier':
        n_estimators = c1.text_input('Number of trees in the forest', '100', help="Number of trees in the forest. Default is 100.").split(',')
        criterion = c2.multiselect('Function to measure split quality', ['gini', 'entropy'], default='gini', help="Function to measure split quality. Default is 'gini'.")
        max_depth = c3.text_input('Maximum depth of the trees', help="Maximum depth of the trees. Default is None.").split(',')
        min_samples_split = c4.text_input('Minimum samples required to split a node', '2', min_value=2, step=1, help="Minimum number of samples required to split an internal node. Default is 2.").split(',')
        n_jobs = c5.text_input('Number of parallel jobs to run', help="Number of parallel jobs to run. Default is None.").split(',')
        parameters = {'n_estimators':n_estimators, 'criterion':criterion, 'max_depth':max_depth, 'min_samples_split':min_samples_split, 'n_jobs':n_jobs}

    elif models_options=='GradientBoostingClassifier':
        learning_rate = c1.text_input('Learning rate', '0.1', help="Learning rate shrinks the contribution of each tree. Default is 0.1.").split(',')
        n_estimators = c2.text_input('Number of boosting stages', '100', help="Number of boosting stages. Default is 100.").split(',')
        max_depth = c3.text_input('Maximum depth of individual estimators', '3', help="Maximum depth of individual regression estimators. Default is 3.").split(',')
        subsample = c4.text_input('Fraction of samples used for fitting', '1.0', help="Fraction of samples used for fitting the individual base learners. Default is 1.0.").split(',')
        parameters = {'learning_rate':learning_rate, 'n_estimators':n_estimators, 'max_depth':max_depth, 'subsample':subsample}
        
    elif models_options=='AdaBoostClassifier': 
        n_estimators = c1.text_input('Number of weak learners (trees)', '50', help="Number of weak learners (trees). Default is 50.").split(',')
        learning_rate = c2.text_input('Learning rate', '1.0', help="Weight applied to each classifier. Default is 1.0.").split(',')
        parameters = {'n_estimators':n_estimators, 'learning_rate':learning_rate}
                
    elif models_options=='XGBClassifier': 
        n_estimators = c1.text_input('Number of boosting rounds', '100', help="Number of boosting rounds. Default is 100.").split(',')
        max_depth = c2.text_input('Maximum depth of a tree', '6', help="Maximum depth of a tree. Default is 6.").split(',')
        learning_rate = c3.text_input('Boosting learning rate', '0.3', help="Boosting learning rate. Default is 0.3.").split(',')
        subsample = c4.text_input('Fraction of samples used for training', '1.0', help="Fraction of samples used for training. Default is 1.0.").split(',')
        parameters = {'n_estimators':n_estimators, 'max_depth':max_depth, 'learning_rate':learning_rate,  'subsample':subsample}
    if custom:
        return parameters
    elif st.button("Submit", type="primary"):
        return parameters
    else:
        return None

def category_reg_params(params, models_options):    
    param = params[models_options]
    c1, c2, c3, c4 = st.columns(4)
    if len([par['Parameter'] for par in param]) > 4:
        c5, c6, c7, c8 = st.columns(4)  
        
    if models_options=='LinearRegression':
        fit_intercept = c1.multiselect('Fit intercept',['True', 'False'],  default='False', help="Whether to calculate the intercept. Default is True.")
        normalize = c2.multiselect('Normalize', ['True', 'False'],  default='False', help="Deprecated; used to normalize before regression. Default is False.")
        copy_X = c3.multiselect('Copy X', ['True', 'False'],  default='False', help="If True, X will be copied; else, overwritten. Default is True.")
        n_jobs = c4.text_input('Number of parallel jobs', 'None', help="Number of parallel jobs to run. Default is None.").split(',')
        parameters = {'fit_intercept':fit_intercept, 'normalize':normalize, 'copy_X':copy_X,  'n_jobs':n_jobs}

    elif models_options=='Lasso':
        alpha = c1.text_input('Constant that multiplies the L1 term', '1.0', help="Constant that multiplies the L1 term. Default is 1.0.").split(',')
        max_iter = c2.text_input('Maximum number of iterations', '1000', help="Maximum number of iterations. Default is 1000.").split(',')
        tol = c3.text_input('Tolerance for optimization', '0.0001', help="Tolerance for the optimization. Default is 0.0001.").split(',')
        parameters = {'alpha':alpha, 'max_iter':max_iter, 'tol':tol}

    elif models_options=='Ridge':
        alpha = c1.text_input('Regularization strength', '1.0', help="Regularization strength. Default is 1.0.").split(',')
        solver = c2.multiselect('Solver for optimization', options=['auto', 'svd', 'cholesky', 'lsqr', 'saga'], default='auto', help="Solver to use in the optimization problem. Default is 'auto'.")
        parameters = {'alpha':alpha, 'solver':solver}

    elif models_options=='ElasticNet':
        alpha = c1.text_input('Alpha', '1.0', help="Constant that multiplies the penalty terms. Default is 1.0.").split(',')
        l1_ratio = c2.text_input('L1 ratio', '0.5', help="The mix ratio between L1 and L2 regularization. Default is 0.5.").split(',')
        max_iter = c3.text_input('Maximum iterations', '1000', help="Maximum number of iterations. Default is 1000.").split(',')
        parameters = {'alpha':alpha, 'l1_ratio':l1_ratio, 'max_iter':max_iter}

    elif models_options=='SVR':
        kernel = c1.multiselect('Kernel type', options=['linear', 'poly', 'rbf', 'sigmoid'], default='rbf', help="Specifies the kernel type to be used.")
        degree = c2.text_input('Degree of polynomial kernel', '3', help="Degree of the polynomial kernel. Default is 3.").split(',')
        C = c3.text_input('Regularization parameter', '1.0', help="Regularization parameter. Default is 1.0.").split(',')
        epsilon = c4.text_input('Epsilon-tube', '0.1', help="Epsilon-tube within which no penalty is given. Default is 0.1.").split(',')
        parameters = {'kernel':kernel, 'degree':degree, 'C':C, 'epsilon':epsilon}

    elif models_options=='KNeighborsRegressor':
        n_neighbors = c1.text_input('Number of neighbors', '5', help="Number of neighbors to use. Default is 5.").split(',')
        weights = c2.multiselect('Weight function', options=['uniform', 'distance'], default='uniform', help="Weight function ('uniform', 'distance'). Default is 'uniform'.")
        algorithm = c3.multiselect('Algorithm for nearest neighbors', options=['auto', 'ball_tree', 'kd_tree', 'brute'], default='auto', help="Algorithm used to compute nearest neighbors. Default is 'auto'.")
        parameters = {'n_neighbors':n_neighbors, 'weights':weights, 'algorithm':algorithm}

    elif models_options=='DecisionTreeRegressor':
        criterion = c1.multiselect('Criterion for splitting', options=['friedman_mse', 'absolute_error', 'squared_error', 'poisson'], help="Function to measure split quality ('mse', 'mae').")
        splitter = c2.multiselect('Splitter for tree', options=['best', 'random'], help="Strategy to split nodes ('best', 'random').")
        max_depth = c3.text_input('Maximum depth of the tree', help="Maximum depth of the tree. Default is None.").split(',')
        min_samples_split = c4.text_input('Minimum samples required to split a node', help="Minimum number of samples required to split an internal node.").split(',')
        parameters = {'criterion':criterion, 'splitter':splitter, 'max_depth':max_depth, 'min_samples_split':min_samples_split}
        
    elif models_options=='RandomForestRegressor':
        n_estimators = c1.text_input('Number of trees in the forest', '100', help="Number of trees in the forest. Default is 100.").split(',')
        criterion = c2.multiselect('Criterion for splitting', options=['mse', 'mae'], default='mse', help="Function to measure split quality. Default is 'mse'.")
        max_depth = c3.text_input('Maximum depth of the trees', 'None', help="Maximum depth of the trees. Default is None.").split(',')
        min_samples_split = c4.text_input('Minimum samples required to split a node', '2', help="Minimum number of samples required to split an internal node. Default is 2.").split(',')
        n_jobs = c5.text_input('Number of parallel jobs to run', 'None', help="Number of parallel jobs to run. Default is None.").split(',')
        parameters = {'n_estimators':n_estimators, 'criterion':criterion, 'max_depth':max_depth, 'min_samples_split':min_samples_split, 'n_jobs':n_jobs}

    elif models_options=='GradientBoostingRegressor':
        learning_rate = c1.text_input('Learning rate', '0.1', help="Shrinks contribution of each tree. Default is 0.1.").split(',')
        n_estimators = c2.text_input('Number of boosting stages', '100', help="Number of boosting stages. Default is 100.").split(',')
        max_depth = c3.text_input('Maximum depth of the individual estimators', '3', help="Maximum depth of the individual estimators. Default is 3.").split(',')
        subsample = c4.text_input('Fraction of samples used for fitting', '1.0', help="Fraction of samples used for fitting the individual base learners. Default is 1.0.").split(',')
        parameters = {'learning_rate':learning_rate, 'n_estimators':n_estimators, 'max_depth':max_depth, 'subsample':subsample}

    elif models_options=='AdaBoostRegressor':
        n_estimators = c1.text_input('Number of weak learners', '50', help="Number of weak learners for AdaBoost. Default is 50.").split(',')
        learning_rate = c2.text_input('Learning rate', '1.0', help="Weight applied to each regressor for AdaBoost. Default is 1.0.").split(',')
        parameters = {'n_estimators':n_estimators, 'learning_rate':learning_rate}
    
    elif models_options=='XGBRegressor':
        n_estimators = c1.text_input('Number of boosting rounds', '100', help="Number of boosting rounds for XGBRegressor. Default is 100.").split(',')
        max_depth = c2.text_input('Maximum depth of a tree', '6', help="Maximum depth of the tree for XGBRegressor. Default is 6.").split(',')
        learning_rate = c3.text_input('Boosting learning rate', '0.3', help="Boosting learning rate for XGBRegressor. Default is 0.3.").split(',')
        subsample = c4.text_input('Fraction of samples used for training', '1.0', help="Fraction of samples for training in XGBRegressor. Default is 1.0.").split(',')
        parameters = {'n_estimators':n_estimators, 'max_depth':max_depth, 'learning_rate':learning_rate,  'subsample':subsample}

    if st.button("Submit", type="primary"):
        return parameters
    else:
        return None

def rearrange_params(params, models_options, parameters):
    param = params[models_options]
    param_names = [par['Parameter'] for par in param]
    param_dict = {}
    for name, listn in parameters.items():
        if name in param_names:
            param_info = param[param_names.index(name)]
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
            except ValueError:
                st.write(f"Error processing {name}. Please check input values.")
            param_dict[name] = mod_params
    filtered_params = {k: v for k, v in param_dict.items() if v}
    combinations = list(product(*filtered_params.values()))
    all_combinations = [dict(zip(filtered_params.keys(), val)) for val in combinations]
    return all_combinations, len(combinations)
    
def category_train(params, datasets, models, category):
    datasets_list = list(datasets.keys())
    models_list = list(models.keys())
    data,mod = st.columns(2)
    datasets_options = data.selectbox('Choose a dataset:', [None] + datasets_list, placeholder='Select dataset')
    models_options = mod.selectbox('Choose a model:', [None] + models_list, placeholder='Select model')
    parameters = None
    if datasets_options:
        trainer = CategoryTrainer(category, datasets[datasets_options], datasets_options)
        trainer.get_data()
        y_true = trainer.y_test
    if models_options:
        if category == 'classification':
            parameters = category_clf_params(params, models_options)
        elif category == 'regression':
            parameters = category_reg_params(params, models_options)
    
    if parameters is not None and datasets_options:
        all_combinations, num_combinations = rearrange_params(params, models_options, parameters)
        rows = (num_combinations + 1) // 2  
        for i in range(rows):
            col = st.columns(2)
            for j in range(2):
                idx = i * 2 + j
                if idx < num_combinations:
                    params_st = all_combinations[idx]
                    title = " | ".join([f"<span style='color:#A6E3E9; font-weight:bold;'>{key.capitalize()}: {value}</span>" for key, value in params_st.items()])
                    col[j].markdown(
                                f"""
                                <div style="text-align:center; border-bottom:2px solid #71C9CE; color:#C4E1F6; font-weight:bold; font-size:17px; padding-bottom:10px; margin-bottom:20px;">
                                    Params: {title}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                    col1 = col[j].columns([0.1,0.3,0.6])
                    with col[j]:
                        results_met, y_pred =  trainer.trainer(params_st,models[models_options]())
                        with col1[1]:
                            for metrics, value in results_met.items():
                                if value != 'N/A':
                                    st.markdown(
                                            f"""
                                            <div style="font-weight:bold; color:#DCD6F7; font-size:17px; margin-bottom:10px;">
                                                {metrics}: <span style="color:#FDCEDF;">{value:.2f}</span>
                                            </div>
                                            """,
                                            unsafe_allow_html=True
                                        )
                                    
                        with col1[2]:
                            if category == 'classification':
                                classification_cov_matrix(y_true, y_pred)
                            else:
                                plot_fitted_vs_original(y_true, y_pred)
                            
    else:
        st.warning("Please select a valid dataset and model to proceed.")
    
