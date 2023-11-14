import PyPDF2
import gradio as gr
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import pickle
import numpy as np
import pandas as pd
import webbrowser

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Load the array into another variable
with open('array.pkl', 'rb') as file:
    job_desc_embeds_final = pickle.load(file)

def compute_cosine_similarity(embedding1, embedding2):
    return cosine_similarity([embedding1, embedding2])[0][1]

def extract_text_from_pdf(file_object, x):
    cv_embed = extract_text_from_pdf1(file_object.name)
    similarities = []

    for i in range(len(job_desc_embeds_final)):
        # Calculating cosine similarity
        sim = compute_cosine_similarity(cv_embed, job_desc_embeds_final[i][0])
        similarities.append(sim)

    job_desc_series = [pd.Series(x) for x in job_desc_embeds_final]
    similarities_series = [pd.Series(x) for x in similarities]

    # Create a DataFrame by concatenating the Pandas Series along columns
    combined_df = pd.concat([pd.DataFrame(job_desc_series), pd.DataFrame(similarities_series)], axis=1)

    # Convert the DataFrame back to a NumPy array
    combined_array = combined_df.to_numpy()

    sorted_arr = sorted(combined_array, key=lambda x: x[2], reverse=True)
    text = ""
    for i in range(x):
        if i == 0:
            text = text + "Top " + str(x) + " occupations: \n"
        text = text + str(i+1) + ". \n" + cut_unnecessary(sorted_arr[i][1]) + "\n"

    return text

def cut_unnecessary(input_string):
    dot_index = input_string.find('.') if ('.' in input_string) or ('\n' in input_string) else 300
    
    cut_index = min(dot_index, 300)
    
    result_string = input_string[:cut_index]
    
    result_string = "\t" + result_string

    return result_string

# Example usage
input_str = "This is a sample string. It has more than 100 characters.\nThis is a new line."
result = cut_unnecessary(input_str)
print(result)


def compute_distilBERT_embedding(text):
    tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=1024)
    with torch.no_grad():
        outputs = model(**tokens)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def adjust_text_length(input_text):
    target_length = 1024

    if len(input_text) < target_length:
        # Add 0 for extra characters
        adjusted_text = input_text + '0' * (target_length - len(input_text))
    elif len(input_text) > target_length:
        # Remove everything after the target length
        adjusted_text = input_text[:target_length]
    else:
        # Length is already equal to the target length
        adjusted_text = input_text

    return adjusted_text

def extract_text_from_pdf1(filepath):
    pdf_reader = PyPDF2.PdfReader(filepath)
    text = ''
    for i in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[i]
        text += page.extract_text()
    text_fin = adjust_text_length(text)
    return compute_distilBERT_embedding(text_fin)

iface = gr.Interface(fn=extract_text_from_pdf, inputs=[gr.File(), gr.Slider(1, 10, value=3, label="Number of jobs", info="Choose between 1 and 10", step=1)], outputs="text")
iface.launch()

