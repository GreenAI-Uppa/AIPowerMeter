## How to contribute to this documentation

It uses [Sphinx](https://www.sphinx-doc.org/en/master/), so first you should install it:

```
pip install sphinx
```

### usage

- Modifiy the .rst files in the `docsrc` folder

- Build the documentation into the `docs` folder
```
sphinx-build -b html docsrc/ docs
```
- open docs/index.html in a web browser to visualise locally the new version

- commit and push `docsrc` and `docs`


And the documentation will be available on the [github page](https://greenai-uppa.github.io/IAPowerMeter/)

### Tips

- Use autodoc to add the docstrings, for instance:
```
.. autofunction:: deep_learning_power_measure.power_measure.experiment.Experiment
```

- Sphynx documentation for the [reStructuredText](https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html) format

- adding sections

Create a file, which header will contain the section title, eg with `usage/quick_start.rst`:
```
Quick start
===========
```
Then, add the file path to the toc in the `index.rst`
```
.. toctree::
   :maxdepth: 2

   usage/quick_start
```

### TODO 

Check if a github action is possible to automate the building process
