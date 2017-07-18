import os
from argparse import ArgumentParser


def makedirs(name):
    """helper function for python 2 and 3 to call os.makedirs()
       avoiding an error if the directory to be created already exists"""

    import os, errno

    try:
        os.makedirs(name)
    except OSError as ex:
        if ex.errno == errno.EEXIST and os.path.isdir(name):
            # ignore existing directory
            pass
        else:
            # a different error happened
            raise


class DefaultConfig(object):
    epochs = 10
    batch_size = 128
    d_embed = 100
    d_proj = 100
    d_hidden = 100
    n_layers = 1
    log_every = 1
    lr = .001
    dev_every = 1000
    save_every = 1000
    dp_ratio = .2
    birnn = 'store_false'
    preserve_case = 'store_false'
    no_projection = 'store_false'
    train_embed = 'store_false'
    gpu = -1
    save_path = 'results'
    data_cache = os.path.join(os.getcwd(), '.data_cache')
    vector_cache = os.path.join(os.getcwd(), '.vector_cache/input_vectors.pt')
    word_vectors = 'glove.42B'
    resume_snapshot = ''
    max_size = 50000
    min_freq = 3


def get_args(config=DefaultConfig()):
    parser = ArgumentParser(description='Deep BoW Project')
    parser.add_argument('--epochs', type=int, default=config.epochs)
    parser.add_argument('--batch_size', type=int, default=config.batch_size)
    parser.add_argument('--d_embed', type=int, default=config.d_embed)
    parser.add_argument('--d_proj', type=int, default=config.d_proj)
    parser.add_argument('--d_hidden', type=int, default=config.d_hidden)
    parser.add_argument('--n_layers', type=int, default=config.n_layers)
    parser.add_argument('--log_every', type=int, default=config.log_every)
    parser.add_argument('--lr', type=float, default=config.lr)
    parser.add_argument('--dev_every', type=int, default=config.dev_every)
    parser.add_argument('--save_every', type=int, default=config.save_every)
    parser.add_argument('--dp_ratio', type=int, default=config.dp_ratio)
    parser.add_argument('--no-bidirectional', action=config.birnn, dest='birnn')
    parser.add_argument('--preserve-case', action=config.preserve_case, dest='lower')
    parser.add_argument('--no-projection', action=config.no_projection, dest='projection')
    parser.add_argument('--train_embed', action=config.train_embed, dest='fix_emb')
    parser.add_argument('--gpu', type=int, default=config.gpu)
    parser.add_argument('--save_path', type=str, default=config.save_path)
    parser.add_argument('--data_cache', type=str, default=config.data_cache)
    parser.add_argument('--vector_cache', type=str, default=config.vector_cache)
    parser.add_argument('--word_vectors', type=str, default=config.word_vectors)
    parser.add_argument('--resume_snapshot', type=str, default=config.resume_snapshot)
    parser.add_argument('--max_size', type=str, default=config.max_size)
    parser.add_argument('--min_freq', type=str, default=config.min_freq)
    args = parser.parse_args()
    return args
