# plasticc-2018

My simple CNN ML model for [Kaggle's PLAsTiCC Astronomical
Classification 2018,](https://www.kaggle.com/c/PLAsTiCC-2018) developed
with PyTorch, a machine learning framework.

# Motivation
This repository shows the first time that I wrote almost all of codes
from scratch (some lines did steal from [another project](https://github.com/ICRAR/rfi_ml)).
Although my model performs bad and I was late for submission, but I have
learnt many things and gained much of experience and I am happy to do so.

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
`flux_err` column to generate new data and added to original data. However,
my simple object ids assignment shows an issue with number overflow, so
I reindexed both training set and training set metadata then assigned
makeshift object ids to be equal index numbers.

```python
np.random.seed(129)
training_data['flux'] += training_data['flux_err']*np.random.uniform(-1.0,1.0)
training_data, training_metadata = training_data.reset_index(drop=True), training_metadata.reset_index(drop=True)
training_metadata['object_id2'] = training_metadata.index
```

## Normalization Data
I created pandas multi-index array to group only **mjd** and **flux**
together into each passband and each object id, ignoring whether we can
detect this object or not (column: **detected**). Since `numpy.array`
and `torch.Tensor` accept on equal-size array, I added zero flux data
to the end of each object and each passband. After that, I normalized
**mjd** with a feature scaling and normalized flux data, with negative
and positive values, with Z-score normalization. These have been done by
each object and these `mjd_mean`, `flux_mean` and `flux_std` have been
kept to be more features of the model. After data re-balance and
normalization completed, the data look like this:

| object_id | passbands | col  | 0        | 1        | ... | 70        | 71        |
|-----------|-----------|------|----------|----------|-----|-----------|-----------|
| 0         | 0         | mjd  | 0.078658 | 0.079747 | ... | 7.991779  | 8.991779  |
|           |           | flux | 0.329796 | 0.412225 | ... | 0.000000  | 0.000000  |
|           | 1         | mjd  | 0.000009 | 0.000000 | ... | 12.999966 | 13.999966 |
|           | ...       |  ... | ...      | ...      | ... | ...       | ...       |


## Normalization Metadata
Since I have kept `mjd_mean`, `flux_mean` and `flux_std` from training
data, I added these column to metadata (which oversampled objects have
been added) and then normalize metadata. Some columns have been
normalized with feature scaling and others have been done with Z-score.
Moreover, I kept the statistical properties of each column and used them
as references for normalizing test set, too.

# Model Summary
For this task, I use convolution neural network (CNN) with batch
normalization 1D to make a loss function to be more stable and descent
faster. Starting with a few 1D convolutions on lightcurve data (mjd and
flux), then data were sent to a neural network. For a few layer until data
were squeezed into small number of unit, then added up more features
from metadata. The next layer need to expand and then succeed by a few
more layers to the end with a softmax layer for classification.

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
