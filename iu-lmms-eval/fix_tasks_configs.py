import os
from ruamel.yaml import YAML

yaml = YAML()
yaml.indent(mapping=2, sequence=4, offset=2)
yaml.preserve_quotes = True  # Preserve quotes if any

def process_yaml_file(file_path):
    with open(file_path, 'r') as file:
        try:
            data = yaml.load(file)
        except Exception as exc:
            print(f"Error parsing {file_path}: {exc}")
            return

    if 'dataset_kwargs' in data:
        dataset_kwargs = data['dataset_kwargs']
        if 'token' in dataset_kwargs:
            del dataset_kwargs['token']

            if not dataset_kwargs:
                del data['dataset_kwargs']

            with open(file_path, 'w') as file:
                yaml.dump(data, file)
            print(f"Updated {file_path}")

def search_and_process_yaml_files(root_directory):
    print(root_directory)
    for dirpath, _, filenames in os.walk(root_directory):
        for filename in filenames:
            if filename.endswith('.yaml'):
                file_path = os.path.join(dirpath, filename)
                process_yaml_file(file_path)

if __name__ == "__main__":
    current_path = os.getcwd()
    root_directory = os.path.join(current_path, 'lmms_eval', 'tasks')
    search_and_process_yaml_files(root_directory)