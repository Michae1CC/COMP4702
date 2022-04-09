# COMP4702 -  Machine Learning

The fold is design to provide python examples to every model and data reduction technique seen in COMP4702. In each example, vectors from the input space are provided as $n \times d$ numpy arrays where $n$ is the number of samples and $d$ is the dimension of the input vectors. For single value outputs, the output vector will be an array of length $n$. For mulitparameter outputs the output vector will have dimensions $n \times k$ where $k$ is the dimension of the output space. Also, class labels will be replaced with integer values ranging from $0$ to $C-1$, where $C$ is the number of classes. Loading datasets via the `load_data` method in `data.py` will guarantee the inputs and outputs will meet this specification. As an example, the iris data set can be loaded as

```python
>>> data, labels = load_data("iris", labels=True)
>>> print(data.shape)   
(150, 4)
>>> print(labels.shape)
(150,)
```

The contents are divided up by weeks with a specific file dedicated to each topic within that week.

## Week 4 - Density Estimation

- [k-NN estimator](main/knn.py)