import os
import glob
import shutil
import importlib.util

local_ragas_path = 'ragas'

def import_ragas_custom():
    spec = importlib.util.find_spec('ragas')
    if spec is None or spec.origin is None:
        raise ImportError("Não foi possível encontrar a biblioteca 'ragas'.")

    library_ragas_path = os.path.dirname(spec.origin)

    if not os.path.exists(library_ragas_path):
        raise FileNotFoundError(f"O diretório da biblioteca 'ragas' não foi encontrado: {library_ragas_path}")

    for root, dirs, files in os.walk(local_ragas_path):
        for file in files:
            local_file_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_file_path, local_ragas_path)
            library_file_path = os.path.join(library_ragas_path, relative_path)
            os.makedirs(os.path.dirname(library_file_path), exist_ok=True)
            shutil.copy2(local_file_path, library_file_path)

    print("Arquivos copiados com sucesso!")

def load_env_variables_from_all_env_files():
    env_files = glob.glob('*.env')
    for file_path in env_files:
        with open(file_path, 'r') as file:
            for line in file:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value