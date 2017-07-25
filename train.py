import os
import time
import glob

import torch
import torch.optim as O
import torch.nn as nn

from utils import makedirs
from data_generator import get_kaggle_sequential_data, get_kaggle_bow_data
from model import SequenceClassifier, BoWClassifier

MODEL_COLLECTION = {
    'sequence': SequenceClassifier,
    'bow': BoWClassifier,
}


def train(config, *args, **kwargs):

    if config.model_type == 'sequence':
        vocabs, iters = get_kaggle_sequential_data(
            batch_size=config.batch_size, device=config.gpu,
            max_size=config.max_size, min_freq=config.min_freq)
    elif config.model_type == 'bow':
        vocabs, iters = get_kaggle_bow_data(
            batch_size=config.batch_size, device=config.gpu,
            max_size=config.max_size, min_freq=config.min_freq)

    vocab = vocabs[0]
    train_iter, valid_iter = iters

    config.n_embed = len(vocab.vocab)
    config.d_out = 1  # regression
    config.n_cells = config.n_layers

    # double the number of cells for bidirectional networks
    if config.birnn:
        config.n_cells *= 2

    set_model = MODEL_COLLECTION[config.model_type]
    model = set_model(config)

    criterion = nn.MSELoss()
    opt = O.Adam(model.parameters(), lr=config.lr)

    iterations = 0
    start = time.time()
    makedirs(config.save_path)

    for epoch in range(config.epochs):
        train_iter.init_epoch()
        for batch_idx, batch in enumerate(train_iter):
            model.train(); opt.zero_grad()

            iterations += 1

            # forward pass
            scores = model(batch)

            # calculate loss of the network output with respect to training labels
            loss = criterion(scores, batch.score)

            # backpropagate and update optimizer learning rate
            loss.backward(); opt.step()

            # checkpoint model periodically
            if iterations % config.save_every == 0:
                snapshot_prefix = os.path.join(config.save_path, 'snapshot')
                snapshot_path = snapshot_prefix + \
                    '_loss_{:.6f}_iter_{}_model.pt'.format(loss.data[0], iterations)
                torch.save(model, snapshot_path)
                for f in glob.glob(snapshot_prefix + '*'):
                    if f != snapshot_path:
                        os.remove(f)

            elif iterations % config.log_every == 0:
                # print progress message
                print('Time: {:>2.0f}, Epochs: {}, Iterations: {}, Loss: {:>2.2f}'.format(
                    time.time()-start, epoch, iterations, loss.data[0]))


if __name__ == '__main__':
    from config import train_template, cli, SmallConfig
    train_template(train)
    cli(obj=SmallConfig)
