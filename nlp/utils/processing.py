import os
import pdfplumber
import re
import stanza
import nltk

to_sub: dict[str, str] = {
    # This set of key value pair will be used with re.sub to clean up the
    # text of I to V in roman numbers, -se, long dashes, vs and replacing
    # places names with just their initials, so that their names don't end
    # up separated in multiple tokens
    r'I\.|I+ |IV| V |-se|—| vs|\d+': ' ',
    'Rio de Janeiro': 'RJ',
    'São Paulo': 'SP',
    'Reino Unido': 'UK',
}

STOPWORDS: list[str] = nltk.corpus.stopwords.words('portuguese')
STOPWORDS.append('ser')

def read_files(path_files: list):
    '''read all pdfs in the indicated location with the substring "texto" in its name'''
    texts = []
    for path_file in path_files:
        temp = pdfplumber.open(path_file)
        t = ''
        for page in temp.pages:
            page = page.extract_text()
            t=t+' '+page
        texts.append(t)
    return texts


def clean_special_characters(texts):
    '''Replaces non-letter characters with spaces in a string list'''
    cleaned_texts_list = []
    for text in texts:        
        for pat, repl in to_sub.items():
            text = re.sub(pat, repl, text)
        cleaned_text = re.sub(u'[^\w ]|[0-9]', ' ', text)        
        cleaned_text = cleaned_text.lower()
        cleaned_texts_list.append(cleaned_text)
    return cleaned_texts_list


def remove_stopwords(texts):
    '''remove stopwords from a list of strings based on the nltk-portuguese module'''
    texts_list = []    
    for text in texts:       
        text_no_stop = [w.strip() for w in text if w.strip() not in STOPWORDS]   
        texts_list.append(text_no_stop)
    return texts_list


def tokenization_lemmatization(pathModelStanza, texts):
    '''uses stanza library to lemmatize and tokenize list of strings in portuguese'''
    nlp = stanza.Pipeline(lang='pt', processors='tokenize,lemma', use_gpu=False)
    text_list = []
    for text in texts:
        doc = nlp(text)
        text_list.append(doc)

    tokens = []
    for i in range(len(text_list)):
        aux = []
        for sent in text_list[i].sentences:
            for word in sent.words:               
                aux.append(word.lemma)
        tokens.append(aux) 
    return tokens


def write_text(texts, path=''):    
    '''writes (txt) a list of strings at the given location'''
    i=1
    for text in texts:        
        path_w = os.path.join(path,'cleaned_text-'+str(i)+'.txt')
        i=i+1
        print(path_w)#exibe os arquivos criados
        with open(path_w, "w") as text_file:
            for word in text:
                text_file.write(f'{word} ')


def pre_processing(path: list, output_path='text', model_path='stanza_models', write=False):
    '''calls read and preprocess functions for nlp'''
    texts = read_files(path)
    cleaned_text = clean_special_characters(texts)
    text_lemma = tokenization_lemmatization(model_path, cleaned_text)
    text_no_stop = remove_stopwords(text_lemma)
    if write:
        write_text(texts=text_no_stop, path=output_path)
    return text_no_stop