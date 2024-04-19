# powershell script to create conda environment and install necessary packages
# run this in conda enabled terminal
$new_py_env = 'asi'
Write-Host $new_py_env

conda create --name $new_py_env --no-default-packages --yes
conda activate $new_py_env

conda info --envs
conda info # make sure you have appropriate active environment
conda list

conda install python=3.10 conda pip pandas --update-all --yes
conda install -c conda-forge scikit-learn --update-all --yes
python -c "import sklearn; sklearn.show_versions()"
pip install opendatasets
conda install -c conda-forge imbalanced-learn --update-all --yes
