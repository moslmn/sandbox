import streamlit as st 
import tempfile
import csvfuncttown as csvft
import torch, sqlite3, random, time, sys, os, re, json, argparse
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import basefunct as bf

st.title("Chat with your CSV data, like a pro 👩‍🏫👨‍🏫")
st.markdown("<h3 style='text-align: center; color: white;'>Built by <a href='https://matmatch.com/'>Matmatch</a></h3>", unsafe_allow_html=True)

# create a button 
init_butt = st.button("Click me to initiate the model")
# create a text area to be updated with the model status
message_board = st.empty()
if init_butt:
    # write the status of the model
    message_board.write("Model is loading...")
    pipe = csvft.init_pipeline()
    # keep the pipe in the session state
    st.session_state['pipe'] = pipe
    # write the status of the model
    message_board.write("Model is ready!")
    message_board.write("Database is ready!")
    chroma_client, sentence_transformer_ef = csvft.init_db()
    st.session_state['chroma_client'] = chroma_client
    st.session_state['sentence_transformer_ef'] = sentence_transformer_ef
    st.write("Hi, I am your data chatbot. Ask me anything about your data...")


uploaded_file = st.sidebar.file_uploader("Upload your CSVs or Json", type=["csv", "json"])

if uploaded_file:

    df, categorical = csvft.safe_load(uploaded_file)
    st.session_state['df'] = df
    st.session_state['categorical'] = categorical
    st.write("Here is the first 5 rows of your data:")
    st.write(df.head())
    # file_path, df, categorical, sentence_transformer_ef):
    collection = csvft.init_collection(st.session_state['chroma_client'], uploaded_file.name, st.session_state['df'], st.session_state['categorical'], st.session_state['sentence_transformer_ef'])
    st.session_state['collection'] = collection

    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello ! Ask me anything about " + uploaded_file.name + " 👩🏽‍🔬"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hallo ! 🙋🏽‍♀️"]
    
    #container for the chat history
    response_container = st.container()
    #container for the user's text input
    container = st.container()

    def conversational_chat_prepj(initial_question, pipe, df,):# categorical, collection):
        alpaca_prep = csvft.Q_prep(initial_question, context= "', '".join(df.columns.tolist()) )
        output = pipe(alpaca_prep, max_new_tokens=500)
        prep_query = csvft.reg_find(output)
        prep_j = json.loads(prep_query)
        #print(prep_j)
        return prep_j
    
    def conversational_chat(prep_j, pipe, df, categorical, collection):
        xoutput = csvft.output_query(prep_query=prep_j['context'], collection= collection, categorical = categorical, N = 20)
        close_words = csvft.close_wrd(prep_j, collection)

        if xoutput:
            User_prompt = csvft.user_prompt(df, categorical)
            alpaca_template = csvft.query_prep(xoutput, prep_j, User_prompt, close_words)
        output = csvft.query_out( pipe, alpaca_template)
        return output
    
    with container:
        with st.form(key='my_form', clear_on_submit=False):
            user_input = st.text_input("Query:", placeholder="Talk to your CSV data here (:", key='input')
            submit_button = st.form_submit_button(label='chat')
            
        if submit_button and user_input:
            #initial_question, pipe, df, categorical, collection):

            prep_j = conversational_chat_prepj(user_input, pipe=st.session_state['pipe'], df=st.session_state['df'],)# collection=st.session_state['collection'], categorical=st.session_state['categorical'])
            st.session_state['prep_j'] = prep_j
            if not np.all([prep_j[i] in ["low", "medium", "high"] for i in ['complexity', 'relevance', 'specificity', 'courtesy']]):
                st.write("The json format is not correct")
            if not prep_j.keys() == {'complexity', 'relevance', 'question', 'context', 'specificity', 'courtesy'}:
                st.write("The json format is not correct")
            if np.any([prep_j[i] == "low" for i in ['relevance', 'specificity', 'courtesy']]):
                st.write("Please rephrase your question")


            else:
                st.write("Initialling the response...")
                st.write("question 🤖: " + prep_j['question'])
                st.write("keywords 👾: " + prep_j['context'])
                output = conversational_chat(st.session_state['prep_j'], pipe=st.session_state['pipe'], df=st.session_state['df'], categorical=st.session_state['categorical'], collection=st.session_state['collection'])
                try:
                    #exc_output = eval(output)
                    st.write("creating the response 🧠 ...: ")
                    output = conversational_chat(prep_j, pipe=st.session_state['pipe'], df=st.session_state['df'], categorical=st.session_state['categorical'], collection=st.session_state['collection'])
                    st.session_state['past'].append(user_input)
                    st.session_state['generated'].append(output)
                    st.write(output)
                    response_container.write(output)
                except :
                    st.write("Something went wrong... wong wong 🦧")

    #if st.session_state['generated']:
    #    with response_container:
    #        for i in range(len(st.session_state['generated'])):
    #            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
    #            message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
