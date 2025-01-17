import pandas as pd
import glob

def combine_csv_files(file_list):
    # Lista para armazenar DataFrames individuais
    dataframes = []
    
    # Iterar sobre a lista de arquivos CSV
    for file in file_list:
        # Ler cada arquivo CSV em um DataFrame
        df = pd.read_csv(file, index_col=0)
        # Adicionar o DataFrame à lista
        dataframes.append(df)
    
    # Concatenar todos os DataFrames em um único DataFrame
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    return combined_df

# Exemplo de uso
file_list = glob.glob('result_gpt4omini_llama3.2_3b*.csv')
print(file_list)
combined_df = combine_csv_files(file_list)


combined_df.to_csv('result_gpt4omini_llama3.2_3b.csv')