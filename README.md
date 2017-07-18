... under construction

# Deep Bag-of-Words
Implementation of deep bag-of-words in PyTorch

## Installation

[Install PyTorch](http://pytorch.org/) following the instructions (I used Python 2.7 on OS X with no CUDA).

Clone the [torchtext git repo](https://github.com/pytorch/text) and then run the following to install `torchtext`:

```
python setup.py install
```

Install the rest of the requirements.

```
pip install -r requirements.txt
```

## Data

Data comes from an old competition on [automatic essay scoring](https://www.kaggle.com/c/asap-aes/data).
Unzip and put it into the `./data` directory.

Training data (`training_set_rel3.tsv`) and validation data (`valid_set.tsv`).