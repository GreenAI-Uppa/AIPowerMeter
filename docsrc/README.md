## Documentation

It uses [Sphynx](https://www.sphinx-doc.org/en/master/), so first you should install it. 


### usage

- Modifiy the .rst files in the `docsrc` folder

- Build the documentation into the `docs` folder
```
sphinx-build -b html docsrc/ docs
```

- commit and push `docsrc` and `docs`


And the documentation will be available on the [github page](https://greenai-uppa.github.io/IAPowerMeter/)

### Tips

Use autodoc to add the docstrings, for instance:
```
.. autofunction:: deep_learning_power_measure.power_measure.experiment.Experiment
```

Sphynx documentation for the [reStructuredText](https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html) format

### TODO 

Check if a github action is possible to automate the building process
