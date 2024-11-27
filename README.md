# prx_notebook

## python environment creation
This repo uses the `conda` tool to manage the python virtual environment. Here are the steps to get started:
- Install miniconda
- Open a conda prompt and go to this 'prx_notebook' folder
- Run `conda env create -f .\prx_notebook_conda.yml`. This will create the environment 'prx_notebook' and install all the packages mentioned in 'prx_notebook_conda.yml'
- Run `conda activate prx_notebook` to activate the virtual environment.
- Run `jupyter notebook` to launch the jupyter notebook server. This will open a browser from which you can launch the notebooks ('.ipynb' extension)

## notebook cleaning before git-pushing
- Run the following command 
    - `conda activate prx_notebook`
    - `nbstripout --install --attributes=.gitattributes`

This will install `nbstripout` ([github](https://github.com/kynan/nbstripout)), a tool that automatically clear all outputs from notebook before committing.