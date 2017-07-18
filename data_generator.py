import os
import six

from sklearn.feature_extraction.text import TfidfVectorizer

import torch
from torchtext.data import Field, Dataset, Example, Iterator
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


def get_kaggle_bow_data(batch_size=32, device=-1,
    max_size=50000, min_freq=3):
    train_df, valid_df = build_data()

    vectorizer = TfidfVectorizer(max_features=max_size, min_df=min_freq)
    train_matrix = vectorizer.fit_transform(train_df.essay)
    valid_matrix = vectorizer.transform(valid_df.essay)


if __name__ == '__main__':
    vocabs, iters = get_kaggle_sequential_data()
    vocab = vocabs[0]
    train_iter, valid_iter = iters
    for batch in train_iter:
        print(batch)
        break
