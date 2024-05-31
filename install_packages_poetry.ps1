
# RECOMMENDED OPTION

## // CONFIGURE PROJECT RUNTIME USING POETRY on WINDOWS

# 1. install python :) e.g. python 3.12
# rest of script assumes:
#   - cpython install
#   - python is on PATH
#   - python launcher is installed - py

# 1 B. if using conda instead of system python
#   - open conda enabled terminal
#   - create conda environmet with required python version
#   - activate conda environment
#   - create .venv (step 3. venv will use conda env python interpreter)

# 2. install poetry
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
poetry --version

# 3. create venv virtual envoronment
python -m venv .venv
#or the non default version
& "$Env:LOCALAPPDATA\Programs\Python\Python311\python.exe" -m venv .venv

# 4. activate venv
# requried step if using a conda envir instead of system python, 
#   otherwise poetry packages will get installed into conda envir instead of .venv envir
./.venv/Scripts/Activate.ps1 

# 5. use poetry to install all packages from pyproject.lock with constraints from poetry.lock
# packages should install into venv
poetry install

# 6. to use env in terminal, activate using venv or poetry
./.venv/Scripts/Activate.ps1    # .venv activator, USE THIS
poetry shell                    # poetry activator, spawns a sub-shell; useless in vscode as will not allow to push lines to REPL

# 7. to use env in vscode: > select python interpreter > '.venv':Poetry

# 8. install package using poetry
poetry add your-package-name # installes and updates pyproject.toml

# 9. update lock file (if installed ok and all scripts worked)
poetry lock


