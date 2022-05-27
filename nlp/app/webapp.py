import streamlit as st
import os
import sys
path = os.getcwd().split('/app')[0]
print(path)
sys.path.append(path)
from utils.dataframe import NLPDataFrame
from utils.processing import pre_processing
from utils.plot_nx import plot_nx
import pandas as pd

st.set_option('deprecation.showPyplotGlobalUse', False)


def show_dataframe(corpus: NLPDataFrame, text, sort=True,
                   by=None, pretty_columns = None, ascending=True) -> None:
    
    st.markdown(f'## {text}')
    df = corpus.df
    if sort and not by:
        raise RuntimeError('Sort True requires column name')
    df = df.sort_values(by=by, ascending=ascending) if sort else df
    if pretty_columns:
        df.columns = pretty_columns
    st.dataframe(df)

    st.download_button('Download', df.to_csv(index=False),
                        file_name='dataframe.csv')


def show_wordcloud(corpus: NLPDataFrame):
    plot_nx(corpus, norm=100000, k=0.48, iterations=80, seed=42)
    st.pyplot()


def main(to_csv: bool=False) -> None:
    ''' The main function of the project, first executes grab_lemmas that
    scrapes for the text, cleans and lemmanizes it and then builds a
    pandas.DataFrame 
    Arg:
        to_csv (bool): If set to True, saves the full dataframe to a csv file
    '''

    st.title("Análise de Texto - NLP")
    st.write("Envie um ou mais arquivos PDF para serem tokenizados, lemmatizados e analisados.")
    pdf_list = st.file_uploader("Upload de arquivo pdf.", type=["pdf"], accept_multiple_files=True)
    st.write("Para resetar a aplicação, remova os arquivos.")
    
    if pdf_list:
        list_tokens: list[list[str]] =  pre_processing(pdf_list)

        corpus = NLPDataFrame(list_tokens, idf_log=True)
        pretty_columns = [
            'Tokens', 'Term Frequency', 'TF Mean', 'Document Frequency',
            'Inverse Document Frequency', 'TF-IDF', 'TF-IDF-MEAN'
        ]
        
        show_dataframe(corpus, 'Métricas', by='tf_idf_mean',
                       pretty_columns=pretty_columns, ascending=False)

        show_wordcloud(corpus)        

if __name__ == '__main__':
    main(to_csv=True)