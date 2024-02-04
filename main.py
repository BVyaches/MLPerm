import json

PATH_TO_NOTEBOOK = r"C:\Users\slava\PycharmProjects\MLPerm\MathAnalysis.ipynb"


def get_notebook(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as notebook:
        return json.load(notebook)


def is_code_cell(cell):
    return cell['cell_type'] == "code"


def get_source_from_code_cell(cell):
    return ''.join(cell['source'])


def save_as_python_file(filename, code):
    with open(f'{filename}.py', 'w', encoding='utf-8') as f:
        f.write(code)


def get_code_cells_content(notebook_cells):
    yield from (
        (i, get_source_from_code_cell(current_cell))
        for i, current_cell in enumerate(notebook_cells, 1)
        if is_code_cell(current_cell)
    )


notebook = get_notebook(PATH_TO_NOTEBOOK)
for filename, code in get_code_cells_content(notebook['cells']):
    save_as_python_file(filename, code)
