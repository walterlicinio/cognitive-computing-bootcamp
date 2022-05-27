from utils.dataframe import NLPDataFrame
from utils.plot_nx import plot_nx
from utils import processing

def main(to_csv: bool=False) -> None:
    ''' The main function of the project, first executes grab_lemmas that
    scrapes for the text, cleans and lemmanizes it and then builds a
    pandas.DataFrame 
    Arg:
        to_csv (bool): If set to True, saves the full dataframe to a csv file
    '''

    list_tokens: list[list[str]] =  processing.pre_processing(
        ['_data/texto1.pdf', '_data/texto2.pdf', '_data/texto3.pdf']
    )
    
    df = NLPDataFrame(list_tokens, idf_log=True)
    
    if to_csv:
        df.to_csv('dataframe.csv', index=False)

    plot_nx(df, norm=100000, k=0.48, iterations=80, seed=222, savefig=True)

if __name__ == '__main__':

    main(to_csv=True)
