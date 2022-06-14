import torch
import os
import streamlit as st
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
from nltk.corpus import wordnet
st.title('Translate')
model_path_en2vi = "Yama/yamavi"
model_path_vi2en = "Yama/yamaen"
@st.cache(allow_output_mutation=True,show_spinner=False)
def load_model(model_path_en2vi,model_path_vi2en):
    tokenizer_en2vi = AutoTokenizer.from_pretrained(model_path_en2vi)
    model_en2vi = AutoModelForSeq2SeqLM.from_pretrained(model_path_en2vi)
    tokenizer_vi2en = AutoTokenizer.from_pretrained(model_path_vi2en)
    model_vi2en = AutoModelForSeq2SeqLM.from_pretrained(model_path_vi2en)      
    return model_en2vi, tokenizer_en2vi,model_vi2en,tokenizer_vi2en
def synonym(input):
    if len(input.split(' ')) == 1:
        synonyms = []
        for syn in wordnet.synsets(input):
            for lemma in syn.lemmas():
                synonyms.append(lemma.name())
        return synonyms[1]   
    else:
        return None
model_entovi , tokenizer_entovi, model_vitoen, tokenizer_vitoen  = load_model(model_path_en2vi,model_path_vi2en) 

languages = st.sidebar.radio("Choose your language!",
     ('English to Vietnamese','Vietnamese to English'),key='languages')
if st.session_state.languages == 'English to Vietnamese':
    input = st.text_input(label='Input your text here', value=  'Hello! My name is TriAn')
    tokenized_txt = tokenizer_entovi(input,return_tensors = 'pt').input_ids
                                        
    outputs = model_entovi.generate(tokenized_txt)
    st.write(tokenizer_entovi.decode(outputs[0], skip_special_tokens = True))
else:
    input = st.text_input(label='Nhập vào đây', value=  'Xin chào! Tôi tên Tri Ân')
    tokenized_txt = tokenizer_vitoen(input,return_tensors = 'pt').input_ids
    outputs = model_vitoen.generate(tokenized_txt)
    st.write(tokenizer_vitoen.decode(outputs[0], skip_special_tokens = True))