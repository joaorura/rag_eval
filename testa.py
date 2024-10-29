# Passo 1: Instalar a biblioteca unidecode
# No terminal, execute:
# pip install unidecode

# Passo 2: Importar a biblioteca unidecode
from unidecode import unidecode

# Passo 3: Criar a função para remover acentos
def remover_acentos(texto):
    return unidecode(texto)

# Exemplo de uso
texto_com_acento = "Olá, como você está?"
texto_sem_acento = remover_acentos(texto_com_acento)
print(texto_sem_acento)  # Saída: "Ola, como voce esta?"