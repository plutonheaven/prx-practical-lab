# prx_notebook

## python environment creation
This repo uses the `conda` tool to manage the python virtual environment. Here are the steps to get started:
- Install miniconda
- Open a conda prompt and go to this 'prx_notebook' folder
- Run `conda env create -f .\prx_notebook_conda.yml`. This will create the environment 'prx_notebook' and install all the packages mentioned in 'prx_notebook_conda.yml'
- Run `conda activate prx_notebook` to activate the virtual environment.
- Run `jupyter notebook` to launch the jupyter notebook server. This will open a browser from which you can launch the notebooks ('.ipynb' extension)

## notebook cleaning before git-pushing
Following this [stackoverflow post](https://stackoverflow.com/questions/28908319/how-to-clear-jupyter-notebooks-output-in-all-cells-from-the-linux-terminal/58004619#58004619):
- add to your local file `.git/config` the following lines:
```
[filter "strip-notebook-output"]
    clean = "jupyter nbconvert --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --to=notebook --stdin --stdout --log-level=ERROR"
```
- Create a `.gitattributes` file in your directory with notebooks, with this content:
```
*.ipynb filter=strip-notebook-output
```
Note that you have to manually modify your local `.git/config` file, while the `.gitattributes` file has been pushed to this repository.