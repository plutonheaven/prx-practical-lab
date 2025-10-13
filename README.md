# prx-practical-lab

## Python environment creation
This repo uses the `uv` tool to manage the python virtual environment. Here are the steps to get started:
- Install `uv` using [these instructions](https://docs.astral.sh/uv/getting-started/installation/)
- Execute `uv sync` in a terminal

This will create a `.venv` folder containing the python virtual environment.

## Running python scripts
To run a python script using the virtual environment, use the command
```
uv run script.py
```

## (deprecated) notebook cleaning before git-pushing
Run the following command `nbstripout --install --attributes=.gitattributes`

This will install `nbstripout` ([github](https://github.com/kynan/nbstripout)), a tool that automatically clear all outputs from notebook before committing.