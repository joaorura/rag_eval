import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Carregar os dados dos arquivos CSV
test1 = pd.read_csv('result_opennai_llama3_1.csv')
test2 = pd.read_csv('result_opennai_llama3_2_3b.csv')

# Adicionar uma coluna para identificar a origem dos dados
test1['Modelos'] = 'Llama 3.1 8b'
test2['Modelos'] = 'Llama 3.2 3b'

# Combinar os dois DataFrames
combined_df = pd.concat([test1, test2])

# Filtrar as métricas desejadas
metrics = [
    "answer_relevancy",
    "answer_correctness",
    "answer_similarity",
    "context_precision",
    "context_recall",
    "context_utilization",
    "context_entity_recall",
    "faithfulness"
]
filtered_df = combined_df[['Modelos'] + metrics]

print(filtered_df)
# Derreter o DataFrame para o formato longo
melted_df = filtered_df.melt(id_vars='Modelos', var_name='Métricas', value_name='Valores')

print(melted_df)
# Criar o gráfico de barras
plt.figure(figsize=(18, 14))
sns.barplot(data=melted_df, x='Métricas', y='Valores', hue='Modelos')
plt.title('Avaliação RAG default com os modelos Llama 3.1 8b e Llama 3.2 3b')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()