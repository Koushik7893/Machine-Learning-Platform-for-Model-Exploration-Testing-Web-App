import streamlit as st
import numpy as np
import pandas as pd
import psycopg2
# import sqlite3
import os
from src.components.ChatBot import AIChatbot
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

# ----------------------------    --        --    ------------------------------
# ----------------------------------------------------------------------------
# from dotenv import load_dotenv
# load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URI")


def connect_sql():
    return psycopg2.connect(DATABASE_URL)


# DB_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "database.db")

# def connect_sql():
#     return sqlite3.connect(DB_PATH)

def fetch_api_key(user_id):
    conn = connect_sql()
    cur = conn.cursor()
    cur.execute("SELECT groq_api_key FROM users WHERE id = %s", (user_id,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row and row[0] else None

def save_api_key(user_id, api_key):
    conn = connect_sql()
    cur = conn.cursor()
    cur.execute("UPDATE users SET groq_api_key = %s WHERE id = %s", (api_key, user_id))
    conn.commit()
    conn.close()     

def get_user_sessions(user_id):
    conn = connect_sql()
    cur = conn.cursor()
    cur.execute("SELECT session_id FROM sessions WHERE user_id = %s ORDER BY created_at", (user_id,))
    sessions = [row[0] for row in cur.fetchall()]
    conn.close()
    return sessions

def create_session(session_id, user_id):
    conn = connect_sql()
    cur = conn.cursor()
    cur.execute("INSERT INTO sessions (user_id, session_id) VALUES (%s, %s)", (user_id, session_id))
    conn.commit()
    conn.close()

def get_chat_history(session_id):
    conn = connect_sql()
    cur = conn.cursor()
    cur.execute("SELECT user_message, bot_response FROM chat_history WHERE session_id = %s ORDER BY timestamp", (session_id,))
    history = cur.fetchall()
    conn.close()
    formatted_history = []
    for chat in history:
        formatted_history.append({"role": "user", "content": chat[0]})
        formatted_history.append({"role": "assistant", "content": chat[1]})
    return formatted_history

def save_chat_response(session_id, user_message, bot_response):
    conn = connect_sql()
    cur = conn.cursor()
    cur.execute("INSERT INTO chat_history (session_id, user_message, bot_response) VALUES (%s, %s, %s)",
                (session_id, user_message, bot_response))
    conn.commit()
    conn.close()

def sidebar(user_id):
    st.sidebar.header("Chat Sessions")
    # Get or update API key
    api_key = fetch_api_key(user_id)
    if not api_key:
        api_key = st.sidebar.text_input("Enter your Groq API key:", type="password")
        if st.sidebar.button("Save API Key"):
            save_api_key(user_id, api_key)
            st.sidebar.success("API Key saved!")
    else:
        st.sidebar.write("API Key is set.")

    # Initialize AIChatbot if API key is available
    chatbot_model = AIChatbot(api_key)

    # Get available sessions for the user
    sessions = get_user_sessions(user_id)
    
    # Add an option to create a new session
    if sessions:
        session_options = sessions + ["Create New Session"]
    else:
        session_options = ["Create New Session"]

    selected_session = st.sidebar.selectbox("Select a session", session_options)
    
    # If "Create New Session" is chosen, allow user to enter a new session ID
    if selected_session == "Create New Session":
        new_session_id = st.sidebar.text_input("Enter new Session ID:")
        if new_session_id:
            if new_session_id in sessions:
                st.sidebar.error("Session already exists, please choose another.")
            else:
                create_session(new_session_id, user_id)
                st.sidebar.success("New session created!")
                selected_session = new_session_id
                st.rerun() 

    st.sidebar.subheader("Chat History")
    if selected_session and selected_session != "No sessions found":
        chat_history = get_chat_history(selected_session)
        for msg in chat_history:
            role = "🧑‍💻 You" if msg["role"] == "user" else "🤖 AI"
            st.sidebar.write(f"**{role}:** {msg['content']}")
    else:
        st.sidebar.info("No chat history available.")

    # Input for sending new message
    question = st.sidebar.text_input("Type your message here...")
    if st.sidebar.button("Send"):
        if question and selected_session and chatbot_model:
            bot_response = chatbot_model.generate_response(question, chat_history)
            save_chat_response(selected_session, question, bot_response)
            st.rerun()


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
user_id = dic.get('session', '')
if user_id:
    sidebar(user_id)
    if page == 'dataset':
        dataset(dic.get('select', ''), dic.get('name', ''), dic.get('types', ''))

    elif page == 'model':
        model(dic.get('select', ''), dic.get('name', ''), dic.get('types', ''))

    elif page == 'category':
        category(dic.get('select', ''), dic.get('name', ''), dic.get('types', ''))
else: 
    st.error("User not logged in! Please log in first.")
    st.stop()



