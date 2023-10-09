# OptPresso

Optpresso is a toy project for trying to apply simply machine learning models to espresso. If you are looking for the old attempt to use CNNs to predict shot times from pictures of the grinds refer to https://github.com/badisa/optpresso/releases/tag/v0.0.1.

## Installation and Initialization

Developed/"tested" using python 3.8, suggested to use the same setup if you want to use it.

```
$ conda create -n optpresso python=3.8
$ pip install -e .
```

To install requirements for testing, development and the notebooks, install the additional requirements.

```
$ pip install -r requirements.txt
```

Run the notebooks
```
$ jupyter notebook notebooks/
```

## TODO

* Make encoding of fields more flexible
* Attempt to use Active Learning for espresso shots


