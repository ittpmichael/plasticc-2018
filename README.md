# plasticc-2018

My ML simple CNN model for [Kaggle's PLAsTiCC Astronomical
Classification 2018,](https://www.kaggle.com/c/PLAsTiCC-2018) developed
with PyTorch as machine learning framework.

# Motivation
This repository shows the first time that I wrote almost all of codes
from scratch (some lines did steal from [another project](https://github.com/ICRAR/rfi_ml)).
Although my model performs bad and I was late for submission, but I have
learnt many things and gained much of experience as I am happy to do so.

# Data Preprocessing
*For one who  does not familiar with this project please see background
information on the project site.* There are two training files:
*training_set.csv* and *training_set_metadata.csv*. Using pandas shows
these classes are highly unbalanced.

```python
metadata['target'].value_counts().sort_index()
Out:
6      151
15     495
16     924
42    1193
52     183
53      30
62     484
64     102
65     981
67     208
88     370
90    2313
92     239
95     175
Name: target, dtype: int64
```

## Oversampling
I used an oversampling technique combine with uncertainty of the
`flux_err` column to generate new data and add to original data. However,
my simple object ids assignment shows an issue with number overflow, so
I reindexed both training set and training set metadata and assigned
makeshift object ids to be equal index numbers.

```python
np.random.seed(129)
training_data['flux'] += training_data['flux_err']*np.random.uniform(-1.0,1.0)
training_data, training_metadata = training_data.reset_index(drop=True), training_metadata.reset_index(drop=True)
training_metadata['object_id2'] = training_metadata.index
```

## Normalization Metadata

# Model Summary

# Files Summary
All files are listed here:

* `preprocessing.py`: A file for first used, contains data normalization,
 data generation for unbalance classes.
* `train.py`: The actual training processes are written here, including
data loading, data grouping, data conversion from `numpy.array` to
`torch.Tensor`, training process for each epoch, a method for adjust
 learning rate and a simple data evaluation.
* `train_plasticc.py`: Main function, contains a class of simple CNN
 model and main method for parsing arguments to be used by `train.py`.
* `./constants/__init__.py`: contains necessarily constants used by
others scripts.

# Framework and tools
## Tools
* [PyCharm](https://www.jetbrains.com/pycharm/)
## Machine Learning Framework
* [PyTorch](https://pytorch.org/)
