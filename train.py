import pdb
import time
import glob

import torch
import torch.optim as O
import torch.nn as nn

from config import get_args, makedirs
from data_generator import get_kaggle_sequential_data
from model import SequenceClassifier

args = get_args()

vocabs, iters = get_kaggle_sequential_data(
    batch_size=args.batch_size, device=args.gpu,
    max_size=args.max_size, min_freq=args.min_freq)
vocab = vocabs[0]
train_iter, valid_iter = iters

config = args
config.n_embed = len(vocab.vocab)
config.d_out = 1  # regression
config.n_cells = config.n_layers

# double the number of cells for bidirectional networks
if config.birnn:
    config.n_cells *= 2

model = SequenceClassifier(config)

criterion = nn.MSELoss()
opt = O.Adam(model.parameters(), lr=args.lr)

iterations = 0
start = time.time()
makedirs(args.save_path)


for epoch in range(args.epochs):
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
        if iterations % args.save_every == 0:
            snapshot_prefix = os.path.join(args.save_path, 'snapshot')
            snapshot_path = snapshot_prefix + \
                '_loss_{:.6f}_iter_{}_model.pt'.format(loss.data[0], iterations)
            torch.save(model, snapshot_path)
            for f in glob.glob(snapshot_prefix + '*'):
                if f != snapshot_path:
                    os.remove(f)

        elif iterations % args.log_every == 0:
            # print progress message
            print('Time: {:>2.0f}, Epochs: {}, Iterations: {}, Loss: {:>2.2f}'.format(
                time.time()-start, epoch, iterations, loss.data[0]))
