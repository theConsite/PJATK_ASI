# powershell script to create conda environment and install necessary packages
# run this in conda enabled terminal
$new_py_env = 'asi'
Write-Host $new_py_env

conda create --name $new_py_env --no-default-packages --yes
conda activate $new_py_env

conda info --envs
conda info # make sure you have appropriate active environment
conda list

# conda install -n $new_py_env python=3.10 conda pip pandas --update-all --yes
# conda install -n $new_py_env -c conda-forge scikit-learn --update-all --yes
# pip install opendatasets
# conda install -n $new_py_env -c conda-forge imbalanced-learn --update-all --yes
# conda install -n $new_py_env ipykernel --update-deps --force-reinstall --yes




conda install -n $new_py_env --yes `
    python=3.10 `
    conda `
    pip `
    pandas `
    geopandas `
    seaborn `
    matplotlib `
    ipykernel
    



conda install -n $new_py_env --yes -c conda-forge  --update-all `
    contextily `
    imbalanced-learn `
    scikit-learn `
    opendatasets


conda install -n $new_py_env --yes -c conda-forge `
    kedro `
    kedro-viz

conda install -n $new_py_env --yes -c conda-forge `
    wandb



python -c "import sklearn; sklearn.show_versions()"
kedro info


# conda remove --name $new_py_env --all --yes
# pycaret
