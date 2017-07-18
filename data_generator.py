import os
import six

from sklearn.feature_extraction.text import TfidfVectorizer

import torch
from torchtext.data import Field, Dataset, Example, Iterator, Batch
import pandas as pd

TRAIN_PATH = 'data/training_set_rel3.tsv'
VALID_PATH = 'data/valid_set.tsv'
TEST_PATH = 'data/test_set.tsv'
VALID_SUBMISSION_PATH = 'data/valid_sample_submission_5_column.csv'

IGNORE_FIELD = None
TEXT_FIELD = Field(
    sequential=True, use_vocab=True,
    init_token='<bos>', eos_token='<eos>', pad_token='<pad>',
    tensor_type=torch.LongTensor)
SCORE_FIELD = Field(
    sequential=False, use_vocab=False,
    preprocessing=lambda x: float(x),
    tensor_type=torch.FloatTensor)

FIELDS = [
    ('essay', TEXT_FIELD),
    ('score', SCORE_FIELD),
]

ESSAY_SET_SCORE_RANGE = {
    1: (2, 12),
    2: (1, 6),
    3: (0, 3),
    4: (0, 3),
    5: (0, 4),
    6: (0, 4),
    7: (0, 30),
    8: (0, 60),
}


def _unicode(text):
    return unicode(text, errors='ignore')


def _normalize(score, essay_set):
    min_score, max_score = ESSAY_SET_SCORE_RANGE[essay_set]
    return float(score) - min_score / max_score


def build_score_dict():
    df = pd.read_csv(VALID_SUBMISSION_PATH)
    score_dict = dict(zip(df.prediction_id, df.predicted_score))
    return score_dict


def build_data():
    train_df = pd.read_csv(TRAIN_PATH, sep='\t')
    train_df['score'] = train_df.apply(
        lambda x: _normalize(x.domain1_score, x.essay_set),
        axis=1)
    train_df['essay'] = train_df['essay'].apply(_unicode)

    valid_score_dict = build_score_dict()

    valid_df = pd.read_csv(VALID_PATH, sep='\t')
    valid_df['score'] = valid_df.apply(
        lambda x: _normalize(
            valid_score_dict[x.domain1_predictionid], x.essay_set),
        axis=1)
    valid_df['essay'] = valid_df['essay'].apply(_unicode)

    # -- load the proper data
    # test_df = pd.read_csv(TEST_PATH, sep='\t')
    return train_df, valid_df


class PandasDataset(Dataset):
    def __init__(self, df, fields, **kwargs):
        field_names = map(lambda x: x[0], fields)

        examples = []
        for i, row in df[field_names].iterrows():
            datum = map(lambda x: row[x], field_names)
            example = Example.fromlist(datum, fields)
            examples.append(example)

        super(PandasDataset, self).__init__(examples, fields, **kwargs)


def get_kaggle_sequential_data(batch_size=32, device=-1,
        max_size=50000, min_freq=3):
    train_df, valid_df = build_data()
    train_dataset = PandasDataset(train_df, FIELDS)
    valid_dataset = PandasDataset(valid_df, FIELDS)

    TEXT_FIELD.build_vocab(train_dataset, max_size=max_size, min_freq=min_freq)
    train_iter, valid_iter = Iterator.splits(
        (train_dataset, valid_dataset), batch_size=batch_size, device=device)

    return (TEXT_FIELD, ), (train_iter, valid_iter)


class BoWBatch(Batch):
    def __init__(self, vectorizer, vectorizer_field,
        data=None, dataset=None, device=None, train=True):
        """Create a Batch from a list of examples."""
        self.vectorizer = vectorizer
        if data is not None:
            self.batch_size = len(data)
            self.dataset = dataset
            self.train = train
            for (name, field) in dataset.fields.items():
                if field is not None:
                    if name == vectorizer_field:
                        text_data = [' '.join(getattr(i, vectorizer_field))
                            for i in data]
                        matrix = self.vectorizer.transform(text_data).toarray()
                        setattr(self, name, matrix)
                    else:
                        setattr(self, name, field.numericalize(
                            field.pad(x.__dict__[name] for x in data),
                            device=device, train=train))


class BoWIterator(Iterator):
    def __init__(self, dataset, vectorizer, batch_size, **kwargs):
        self.vectorizer = vectorizer
        super(BoWIterator, self).__init__(dataset, batch_size, **kwargs)

    def __iter__(self):
        while True:
            self.init_epoch()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                yield BoWBatch(
                    self.vectorizer, 'essay',
                    minibatch, self.dataset, self.device,
                    self.train)
            if not self.repeat:
                raise StopIteration

    @classmethod
    def splits(cls, datasets, vectorizer, batch_sizes=None, **kwargs):
        if batch_sizes is None:
            batch_sizes = [kwargs.pop('batch_size')] * len(datasets)
        ret = []
        for i in range(len(datasets)):
            train = i == 0
            ret.append(cls(
                datasets[i], vectorizer,
                batch_size=batch_sizes[i], train=train, **kwargs))
        return tuple(ret)


def get_kaggle_bow_data(batch_size=32, device=-1,
    max_size=50000, min_freq=3):
    train_df, valid_df = build_data()

    train_dataset = PandasDataset(train_df, FIELDS)
    valid_dataset = PandasDataset(valid_df, FIELDS)

    vectorizer = TfidfVectorizer(
        max_features=max_size, min_df=min_freq)
    vectorizer.fit(train_df.essay)

    train_iter, valid_iter = BoWIterator.splits(
        (train_dataset, valid_dataset), vectorizer,
        batch_size=batch_size, device=device)

    return (vectorizer, ), (train_iter, valid_iter)


if __name__ == '__main__':
    vocabs, iters = get_kaggle_sequential_data()
    vocab = vocabs[0]
    train_iter, valid_iter = iters
    for batch in train_iter:
        print(batch)
        break

    vocabs, iters = get_kaggle_bow_data()
    train_iter, valid_iter = iters
    for batch in train_iter:
        print(batch)
        break
