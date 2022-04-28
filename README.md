# COMP4702 - Machine Learning

The repository is designed to provide python examples to every model and data reduction technique seen throughout COMP4702. In each example, vectors from the input space are provided as $n \times d$ numpy arrays where $n$ is the number of samples and $d$ is the dimension of the input vectors. For single value outputs, the output vector will be an array of length $n$. For mulitparameter outputs the output vector will have dimensions $n \times k$ where $k$ is the dimension of the output space. Also, class labels will be replaced with integer values ranging from $0$ to $C-1$, where $C$ is the number of classes. Loading datasets via the `load_data` method in `data.py` will guarantee the inputs and outputs will meet this specification. As an example, the iris data set can be loaded as

```python
>>> data, labels = load_data("iris", labels=True)
>>> print(data.shape)
(150, 4)
>>> print(labels.shape)
(150,)
```

## Topics

The content listed by weeks with a specific file dedicated to each topic.

### Week 2 - Principles of Supervised Learning

- [Logistic Regression](main/log_reg.py)
- [Linear Discriminant Analysis](main/lda.py)
- [Logistic Regression](main/log_reg.py)

### Week 4 - Density Estimation

- [k-NN estimator](main/knn.py)
- [ksd estimation](main/kde.py)

### Week 5 - Clustering

- [k-means clustering](main/k_means.py)
- [Mean Shift](main/mean_shift.py)

### Week 6 - Dimensionality Reduction

- [PCA](main/pca.py)
- [LDA](main/lda.py)
- [t-Stochastic Neighbour Embedding](main/tsne.py)
- [Isomap](main/isomap.py)

### Week 7-9 - Neural Networks

- [Neural Networks](main/nn.py)
- [Auto-Encoding](main/ae.py)

### Week 12 - Exam Pratice

- Try your hand at applying these models to past exam data sets.

## Setup

This assumes you have [git](https://git-scm.com/downloads), [python3](https://www.python.org/downloads/) and [pip3](https://pip.pypa.io/en/stable/cli/pip_download/) installed. To start navigate to a folder you would like to download this repo and run:

```bash
git clone https://github.com/Michae1CC/COMP4702.git COMP4702-examples
```

You should find a newly created folder called `COMP4702-examples`. Change into this folder by running `cd COMP4702-examples`. Next download all the required packages by running:

```bash
pip3 install -r requirements.txt
```

[PyTorch](https://pytorch.org/get-started/locally/) will need to be installed through their website. You should now be able to run any example with the main folder:

```bash
python3 main/k_means.py
```
