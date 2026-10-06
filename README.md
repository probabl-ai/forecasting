# Tutorial on Forecasting with Machine Learning

## Browse the contents online

- [![Launch JupyterLite](./book/images/jupyterbook_badge.svg 'Our JupyterBook
website')](https://probabl-ai.github.io/forecasting/)
Static HTML with precomputed results.

- [![Launch JupyterLite](./book/images/jupyterlite_badge.svg 'Our JupyterLite
website')](https://probabl-ai.github.io/forecasting/jupyterlite/lab)
In-browser interactive execution environment, no install needed.

## Local setup

### Install `pixi`

You can refer to the [official website](https://pixi.sh/latest/#installation) for
installation.

### Launching Jupyter Lab

To launch Jupyter Lab, run the following command:

```bash
pixi run jupyter lab
```

The Python environment and necessary packages will be automatically installed for you.

### Opening lecture notes

The lecture notes are available in the `content/python_files` directory. To open the
Python file as notebook, you need to right click on the file and select `Open with` ->
`Notebook`.

Alternatively, you can generate notebooks as well:

```bash
pixi run convert-to-notebooks
```

This will convert the Python files into notebooks in the folder `content/notebooks`.

### Building and testing the jupyterlite deployment locally

Test the deployment locally:

```bash
pixi run serve-jupyterlite
```

Then open http://[::]:8000/ in a web browser.

Note: firefox often complains about Service Workers not being initialized when
running jupyterlite served from localhost. This prevents accessing the local
dataset files. Use a different browser to test the local deployment.
