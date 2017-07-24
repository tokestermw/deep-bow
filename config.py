import os
import functools

import click


class DefaultConfig(object):
    log_every = 10
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
    birnn = False
    preserve_case = False
    no_projection = False
    fix_emb = False
    projection = False
    gpu = -1
    save_path = 'results'
    data_cache = os.path.join(os.getcwd(), '.data_cache')
    vector_cache = os.path.join(os.getcwd(), '.vector_cache/input_vectors.pt')
    word_vectors = 'glove.42B'
    resume_snapshot = ''
    max_size = 50000
    min_freq = 3


class SmallConfig(DefaultConfig):
    log_every = 1
    d_embed = 25
    d_proj = 25
    d_hidden = 25
    save_every = 100
    max_size = 5000


@click.group()
@click.pass_context
def cli(context):
    if context.obj is None:
        context.obj = DefaultConfig()


def override_context(f):
    @click.pass_context
    def new_func(context, *args, **kwargs):
        for k in kwargs:
            v = kwargs[k]
            if v is None:
                click.echo('Ignored command line argument: {}'.format(k))
            else:
                setattr(context.obj, k, v)
        return context.invoke(f, context.obj, *args, **kwargs)
    return functools.update_wrapper(new_func, f)


def train_template(f):
    @cli.command(context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    ))
    @click.option('--epochs', type=int)
    @click.option('--batch_size', type=int)
    @click.option('--d_embed', type=int)
    @click.option('--d_proj', type=int)
    @click.option('--d_hidden', type=int)
    @click.option('--n_layers', type=int)
    @click.option('--log_every', type=int)
    @click.option('--lr', type=float)
    @click.option('--dev_every', type=int)
    @click.option('--save_every', type=int)
    @click.option('--dp_ratio', type=float)
    @click.option('--birnn', type=bool, is_flag=True)
    @click.option('--preserve_case', type=bool, is_flag=True)
    @click.option('--no_projection', type=bool, is_flag=True)
    @click.option('--fix_emb', type=bool, is_flag=True)
    @click.option('--projection', type=bool, is_flag=True)
    @click.option('--gpu', type=int)
    @click.option('--save_path', type=click.STRING)
    @click.option('--data_cache', type=click.STRING)
    @click.option('--vector_cache', type=click.STRING)
    @click.option('--word_vectors', type=click.STRING)
    @click.option('--resume_snapshot', type=click.STRING)
    @click.option('--max-size', type=int)
    @click.option('--min-freq', type=int)
    @override_context
    @click.pass_obj
    def train(config, *args, **kwargs):
        return f(config, *args, **kwargs)

    return train
