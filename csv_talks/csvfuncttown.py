import numpy as np
import streamlit as st
import pandas as pd
import tqdm, re, json
from collections import Counter
import matplotlib.pyplot as plt
import transformers
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import chromadb
from chromadb.utils import embedding_functions
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction as dd
from scipy.spatial import distance_matrix



# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16

)

@st.cache_resource()
def init_pipeline():
    model_id = "Phind/Phind-CodeLlama-34B-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_id, legacy=True)
    model = AutoModelForCausalLM.from_pretrained(model_id , quantization_config=bnb_config, )
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, 
                    max_new_tokens=500,  
                    trust_remote_code=True,
                    #device='cuda', 
                    return_full_text=True,
                    pad_token_id=tokenizer.eos_token_id
                    )

    pipe.tokenizer.pad_token_id = model.config.eos_token_id
    return pipe

@st.cache_resource()
def init_db():
    #default_ef = embedding_functions.DefaultEmbeddingFunction()
    sentence_transformer_ef = dd(model_name="all-roberta-large-v1")
    chroma_client = chromadb.PersistentClient(path="test_chroma_webapp.db")
    return chroma_client, sentence_transformer_ef

def init_collection(chroma_client, file_path, df, categorical, sentence_transformer_ef):

    try:
        collection = chroma_client.get_collection(file_path, embedding_function=sentence_transformer_ef)
        print(f"Colleciton {file_path} is loaded")
    except:
        collection = chroma_client.create_collection(file_path, 
                                                    embedding_function=sentence_transformer_ef,
                                                    metadata={"hnsw:space": "cosine"})
        for col in df.columns:
            docx = col
            embs = sentence_transformer_ef([docx])
            collection.add(
                documents= [docx],
                embeddings=embs,
                # add ids to the documents
                metadatas= [{"category": "column"}],
                ids= [f"{col}"],
            )
        for cat in tqdm.tqdm(categorical, desc="Loading data into ChromaDB"):
            docx = df[cat].unique().tolist()
            embs = sentence_transformer_ef(docx)
            collection.add(
                documents= docx,
                embeddings=embs,
                # add ids to the documents
                metadatas= [{"category": cat} for i in range(len(docx))],
                ids= [f"{cat}-{i}" for i in range(len(docx))],
            )
    return collection

def apply_chat_template(chat):
  """
  Converts a list of chat messages to the Alpaca template format.

  Args:
    chat: A list of dictionaries, where each dictionary represents a chat message
          with keys "role" and "content".

  Returns:
    A string containing the chat data in the Alpaca template format.
  """

  template = ""
  for message in chat:
    role = message["role"]
    content = message["content"]

    if role == "user":
      template += f"\n### User Message\n{content}"
    elif role == "system prompt":
      template += f"\n### System Prompt\n{content}"
    else:
      print(f"Warning: Unknown role '{role}' in chat data.")
  return template

def reg_find(output):
    try:
        text_block = output[0]["generated_text"]
    except:
        text_block = output
    x = re.findall(r"(?<=###)(.*?)(?=$|###)", text_block, flags=re.DOTALL)
    text = x[-1]

    # regex for text between <pre> and </pre>
    x = re.findall(r"(?<=<pre>)(.*?)(?=<\/pre>)", text, flags=re.DOTALL)
    return x[0].strip()
    
def weird_number(x):
    #regex for the first number, int or float
    number = re.compile(r"[-+]?\d*\.\d+|\d+")
    return number.findall(x)[0]

def safe_load(file_path):
    # find the file extension
    ext = file_path.name.split(".")[-1]
    # call the right function of pd.read_*
    ext = ext.lower()
    pdext = getattr(pd, f'read_{ext}')
    df = pdext(file_path)
    for clm in df.columns:
        try:
            df[clm] = df[clm].apply(weird_number)
            df[clm] = pd.to_numeric(df[clm] , errors='raise').astype(float)
        except Exception as e:
            pass
    # convert True/False to Yes/No
    df = df.replace({True: "Yes", False: "No"})
    # drop rows with NaN values
    df = df.dropna()
    categorical = list(set(df.columns) - set(df.describe().columns))
    return df, categorical

# most common words in df[categorical]
def most_common_words(df, columns, N):
    """
    Finds the most common words in a DataFrame's columns.
    Args:
      df: A pandas DataFrame.
      columns: A list of columns to analyze.
    Returns:
      A list of the most common words in the columns.
    """
    words = df[columns].unique() 
    return Counter(words).most_common(N)

def pd_column_text_summary(df, column):
    """
    Summarizes a column in a DataFrame using the `describe` method and returns a text block.
    Args:
      df: A pandas DataFrame.
      column: The column to summarize.
    Returns:
      A text block summarizing the column.
    """
    summary = df[column].describe()
    text_block = f""""{column}" has a mean of {round(summary["mean"], 2)}, a standard deviation of {round(summary["std"], 2)}, a minimum of {round(summary["min"], 2)}, and a maximum of {round(summary["max"], 2)}.""" 
    return text_block

def pd_column_text_summary_categorical(df, column):
    """
    Summarizes a categorical column in a DataFrame and returns a text block.
    Args:
      df: A pandas DataFrame.
      column: The column to summarize.
    Returns:
      A text block summarizing the column.
    """
    cat_summary = {}
    #cat_summary[column] = [i[0] for i in most_common_words(df, columns=column, N=10)]
    tmp = []
    for i in most_common_words(df, columns=column, N=10):
        tmp.append(i[0])
        if len('", "'.join(tmp)) > 100:
            break
    cat_summary[column] = tmp
    text_block = f""""{column}" is about "{'", "'.join(cat_summary[column])}", etc."""
    return text_block

def output_query(prep_query, collection, categorical, N = 20):
  prep_query = ' '.join(prep_query)
  xoutput = []
  results = collection.query(
                              query_texts=[prep_query],
                              n_results=N,
                              #where={"category": {"$in": [cat]} }
                                      ) 

  x = np.array(results['documents'][0])
  y = np.array(results['distances'][0])
  mean = np.mean(y)
  std = np.std(y)
  ans = x[y < mean ]
  cnt = 0 
  xoutput.append([std, mean, 'all', len(ans), "'" + "', '".join(ans.tolist())+ "'"])
  
  for poss_col in prep_query.split(", "):
      results = collection.query(
                                query_texts=[poss_col],
                                n_results=N,
                                where={"category": {"$eq": "column"} }
                                      ) 
      x = np.array(results['documents'][0])
      y = np.array(results['distances'][0])
      ans = x[:3]
      cat = "C " + results['ids'][0][0]
      mean = np.mean(y)
      std = np.std(y)
      xoutput.append([std, mean, cat, len(x), "'" + "', '".join(ans.tolist())+ "'"])
  
  for cat in tqdm.tqdm(categorical, desc="Querying ChromaDB"):
      cnt = 0
      ans = []
      while len(ans) < 10 and cnt < 10:
          cnt += 1
          results = collection.query(
                                      query_texts=[prep_query],
                                      n_results=N+cnt,
                                      where={"category": {"$in": [cat]} }
                                      )  
          x = np.array(results['documents'][0])
          y = np.array(results['distances'][0])
          mean = np.mean(y)
          std = np.std(y)
          ans = x[y < mean ]
  
      xoutput.append([std, mean, cat, len(ans), ans])#[cat, "'" + "', '".join(ans.tolist())+ "'"])

  return xoutput


def close_wrd(prep_j, collection):
    close_words = []
    #print(prep_j['context'])
    for item in prep_j['context'].split(", "):
        tmp = []
        results = collection.query(
            query_texts=[item],
            n_results=10,
            #where={"category": {"$eq": "column"} }
            )
        #print(item, results["documents"][0][:3])
        for j in results["documents"][0][:3]:
            if item.lower() not in j.lower():
                tmp.append(j)
        if len(tmp) > 0:
            close_words.append([item, ", ".join(tmp)])
    return close_words

def user_prompt(df, categorical):
    user_prompt = ""
    for column in df.columns:
        if column not in categorical:
            #print(pd_column_text_summary(df, column))
            user_prompt += "The column " + pd_column_text_summary(df, column) + "\n"
        else:
            #print(pd_column_text_summary_categorical(df, column))
            user_prompt += "The categorical column " + pd_column_text_summary_categorical(df, column) + "\n"

    #print(user_prompt)
    return user_prompt

# Example usage
chat_data = [
  {"role": "system prompt", "content": """Assistant is a expert assistant that respond to the question without additional words or explanations."""},
  {"role": "user", "content": ''},
]

def Q_maker(question, rel_col_name, context, user_prompt = user_prompt , chat_data = chat_data, query_add_info = None ):
    Q = f"""These are the inofrmation about a pandas DataFrame, named df:\n{user_prompt}
    - Based on these columns, answer the following question by create corresponding pandas queries.
    Follow theses guidelines:
    - Try avoidng using .head() methods.
    - start the answer by using <pre> and finish by </pre>.
    - Use only the column names from the dataset.
    - Instead of using == for queries, use .str.contains(), or relative operators, like > and <.
    - If the Question complexity is high or medium, use broader queries, with less specific conditions.    
    Question complexity:
    {query_add_info}
    Question:
    {question}
    If you need more context:
    {context}
    Relevant column in order of importance:
    '{rel_col_name}'
    ### Assistant
    <pre>"""
    chat_data[-1]["content"] = Q
    alpaca_template = apply_chat_template(chat_data)

    return alpaca_template


def Q_prep(question, context , chat_data = chat_data):
    Q = f"""
    This is a pandas DataFrame, with following columns:\n'{context}'
    Given the Main Question below, do the following steps:
    - Relevance to the dataset: determine if the question is relevant to the dataset. Use high, medium, or low.
    - Complexity of the question: determine if the question is complex or simple. Use high, medium, or low.
    - Specifity of the question: determine if the question is specific, with respect to the dataframe columns. Use high, medium, or low.
    - Courtesy and politeness of the question: determine if the question is polite and respectful. Use high, medium, or low.
    - Question: correct the grammer errors and abbreviations. rewrite the question in correct form.
    - Context of the question: extract the key words from the question.
    and put them in a json format. 
    For example:
    User Question: "What is the average of column A?"
    {{"relevance": "high",
    "complexity": "low",
    "specificity": "high",
    "courtesy": "high",
    "question": "what is the average of column A?",
    "context": "average, column A"
    }}
    Main Question:
    {question.lower()}?
    ### Assistant
    <pre>{{"relevance": """
    chat_data[-1]["content"] = Q
    alpaca_template = apply_chat_template(chat_data)

    return alpaca_template

def query_prep(xoutput, prep_j, user_prompt, close_words):
    xop = []
    col_cat = []
    for i in range(len(xoutput)):
        if xoutput[i][3]>0:
            print(xoutput[i])
            xop.append([xoutput[i][0], xoutput[i][1] ])
            col_cat.append(xoutput[i][2])

    nxop = np.array(xop)
    # distance matrix
    dist = distance_matrix(nxop, nxop)
    rel_col_name = np.array(col_cat)[np.argsort(dist[0])[1:]]
    edited = []
    for item in rel_col_name.tolist():
        if not item.startswith("C "):
            edited.append(item)
        else:
            if item[2:] not in edited:
                edited.append(item[2:])
    rel_col_name = edited.copy()
    conny = "".join(prep_j['context'])
    conny += "\n"
    for item in close_words:
        conny += f""""{item[0]}" is similar to these words: "{item[1]}. """
    
    alpaca_template = Q_maker(question=prep_j['question'], rel_col_name= "', '".join(rel_col_name), 
                            context = conny, user_prompt = user_prompt, query_add_info = prep_j['complexity'])
    
    return alpaca_template

def query_out(pipe, alpaca_template):

    output = pipe(alpaca_template, max_new_tokens=500)
    #print(output[0]["generated_text"].split("###")[-1])
    command = reg_find(output)

    cmd = [i.strip() for i in command.split('\n')]
    cmd = list(filter(None, cmd))
    
    return cmd
        

