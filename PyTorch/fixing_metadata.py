import nbformat

# Load notebook
notebook_path = "04_pytorch_custom_datasets.ipynb"  # Change this
nb = nbformat.read(notebook_path, as_version=4)

# Fix or remove invalid widget metadata
if 'widgets' in nb['metadata']:
    if 'state' not in nb['metadata']['widgets']:
        del nb['metadata']['widgets']

# Save fixed notebook
fixed_path = "03_pytorch_custom_datasets_fixed.ipynb"
nbformat.write(nb, fixed_path)
print(f"Fixed notebook saved as {fixed_path}")
