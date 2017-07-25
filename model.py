import torch
import torch.nn as nn
from torch.autograd import Variable


class Bottle(nn.Module):
    """Time distributed dense.
    (assumes first dimension is time, second dimension is batch_size)
    """
    def forward(self, input):
        if len(input.size()) <= 2:
            return super(Bottle, self).forward(input)
        size = input.size()[:2]
        out = super(Bottle, self).forward(input.view(size[0]*size[1], -1))
        return out.view(size[0], size[1], -1)


class Linear(Bottle, nn.Linear):
    pass


class SequenceEncoder(nn.Module):

    def __init__(self, config):
        super(SequenceEncoder, self).__init__()
        self.config = config
        input_size = config.d_proj if config.projection else config.d_embed
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=config.d_hidden,
                        num_layers=config.n_layers, dropout=config.dp_ratio,
                        bidirectional=config.birnn)

    def forward(self, inputs):
        batch_size = inputs.size()[1]
        state_shape = self.config.n_cells, batch_size, self.config.d_hidden
        h0 = c0 = Variable(inputs.data.new(*state_shape).zero_())
        outputs, (ht, ct) = self.rnn(inputs, (h0, c0))

        if self.config.birnn:
            return ht[-2:].transpose(0, 1).contiguous().view(batch_size, -1)
        else:
            return ht[-1]


class SequenceClassifier(nn.Module):
    
    def __init__(self, config):        
        super(SequenceClassifier, self).__init__()
        self.config = config
        self.embed = nn.Embedding(config.n_embed, config.d_embed)
        if self.config.projection:
            self.projection = Linear(config.d_embed, config.d_proj)
        self.encoder = SequenceEncoder(config)
        self.dropout = nn.Dropout(p=config.dp_ratio)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        seq_in_size = config.d_hidden
        if self.config.birnn:
            seq_in_size *= 2
        lin_config = [seq_in_size]*2
        # TODO: make this depth configurable
        self.output = nn.Sequential(
            Linear(*lin_config),
            self.relu,
            self.dropout,
            Linear(*lin_config),
            self.relu,
            self.dropout,
            Linear(*lin_config),
            self.relu,
            self.dropout,
            Linear(seq_in_size, config.d_out),
            self.sigmoid)

    def forward(self, batch):
        embedding = self.embed(batch.essay)
        if self.config.fix_emb:
            embedding = Variable(embedding.data)
        if self.config.projection:
            embedding = self.relu(self.projection(embedding))
        essay_vector = self.encoder(embedding)
        scores = self.output(essay_vector)
        return scores


class BoWEncoder(nn.Module):

    def __init__(self, config):
        super(BoWEncoder, self).__init__()
        self.config = config
        input_size = config.max_size
        self.projection = Linear(input_size, config.d_hidden)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        batch_size = inputs.size()[0]
        out = self.projection(inputs)
        out = self.relu(out)
        return out


class BoWClassifier(nn.Module):
    def __init__(self, config):
        super(BoWClassifier, self).__init__()
        self.config = config
        self.encoder = BoWEncoder(config)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=config.dp_ratio)
        self.sigmoid = nn.Sigmoid()
        seq_in_size = config.d_hidden
        lin_config = [seq_in_size]*2
        self.output = nn.Sequential(
            Linear(*lin_config),
            self.relu,
            self.dropout,
            Linear(*lin_config),
            self.relu,
            self.dropout,
            Linear(*lin_config),
            self.relu,
            self.dropout,
            Linear(seq_in_size, config.d_out),
            self.sigmoid)

    def forward(self, batch):
        essay_vector = self.encoder(batch.essay)
        scores = self.output(essay_vector)
        return scores
