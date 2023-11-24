import nbformat
from nbconvert import PythonExporter

def clean_script_line(line):
    if line.startswith('#!') or line.startswith('# coding:'):
        return False
    if line.strip().startswith('# In['):
        return False
    return True

def notebook_to_script(notebook_path, output_path):
    # Load the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    # Convert to Python script
    python_exporter = PythonExporter()
    python_exporter.exclude_markdown = True
    python_exporter.exclude_output = True
    
    (body, _) = python_exporter.from_notebook_node(nb)
    
    # Clean script lines
    clean_lines = filter(clean_script_line, body.splitlines())

    # Write to output file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(clean_lines))

notebook_to_script('burgers_pinn.ipynb', 'clean.py')