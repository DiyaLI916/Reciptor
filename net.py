import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data

from args import get_parser
from model.blocks import *
import gensim

# ================================================================================
parser = get_parser()
opts = parser.parse_args()
# # ==============================================================================

# normalization
def norm(input, p=1, dim=1, eps=1e-12):
    return input / input.norm(p, dim, keepdim=True).clamp(min=eps).expand_as(input)

# Skip-thoughts LSTM for handling instructions
class stRNN(nn.Module):
    def __init__(self):
        super(stRNN, self).__init__()
        # keep the same embedding
        self.lstm = nn.LSTM(input_size=opts.stDim, hidden_size=opts.srnnDim, bidirectional=False, batch_first=True)
        self.irnn = nn.LSTM(input_size=opts.stDim, hidden_size=opts.srnnDim, bidirectional=True, batch_first=True)

    def forward(self, x, sq_lengths):
        # here we use a previous LSTM to get the representation of each instruction
        # sort sequence according to the length

        sorted_len, sorted_idx = sq_lengths.sort(0, descending=True)
        # print('sq_lengths', sq_lengths)
        # print('sorted_len, sorted_idx', sorted_len, sorted_idx)

        index_sorted_idx = sorted_idx.view(-1, 1, 1).expand_as(x)
        # print('index_sorted_idx', index_sorted_idx, index_sorted_idx.shape)

        sorted_inputs = x.gather(0, index_sorted_idx.long())
        # print('sorted_inputs', sorted_inputs.shape, sorted_inputs)

        # pack sequence
        packed_seq = torch.nn.utils.rnn.pack_padded_sequence(
            sorted_inputs, sorted_len.cpu().data.numpy(), batch_first=True)

        # pass it to the lstm, didn't use hidden here
        out, hidden = self.lstm(packed_seq)
        # unsort the output
        _, original_idx = sorted_idx.sort(0, descending=False)

        unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        unsorted_idx = original_idx.view(-1, 1, 1).expand_as(unpacked)

        # we get the last index of each sequence in the batch
        idx = (sq_lengths - 1).view(-1, 1).expand(unpacked.size(0), unpacked.size(2)).unsqueeze(1)
        # print(idx)

        # we sort and get the last element of each sequence
        # [n steps, lstm the get the last step have the former n-1 steps info, thus only get the last one]
        output = unpacked.gather(0, unsorted_idx.long()).gather(1, idx.long())
        output = output.view(output.size(0), output.size(1) * output.size(2))

        return output

# Model Ingredients
class ingRNN(nn.Module):
    def __init__(self):
        super(ingRNN, self).__init__()
        self.irnn = nn.LSTM(input_size=opts.ingrW2VDim, hidden_size=opts.irnnDim, bidirectional=True, batch_first=True)

        # _, vec = torchwordemb.load_word2vec_bin(opts.ingrW2V)
        w2v_model = gensim.models.KeyedVectors.load_word2vec_format(opts.ingrW2V, binary=True)
        vec = torch.FloatTensor(w2v_model.vectors)

        self.embs = nn.Embedding(vec.size(0), opts.ingrW2VDim, padding_idx=0)  # not sure about the padding idx
        # padding: </s>, first one in word2vec dict

        self.embs.weight.data.copy_(vec)

    def forward(self, x, sq_lengths):
        # we get the w2v for each element of the ingredient sequence
        x = self.embs(x)
        # print('self.embs(x)', x)

        # sort sequence according to the length
        sorted_len, sorted_idx = sq_lengths.sort(0, descending=True)
        index_sorted_idx = sorted_idx.view(-1, 1, 1).expand_as(x)
        sorted_inputs = x.gather(0, index_sorted_idx.long())

        # pack sequence
        packed_seq = torch.nn.utils.rnn.pack_padded_sequence(
            sorted_inputs, sorted_len.cpu().data.numpy(), batch_first=True)

        # pass it to the rnn
        # output(seq_len, batch, hidden_size * num_directions)
        # h_n(num_layers * num_directions, batch, hidden_size)

        out, hidden = self.irnn(packed_seq)
        # print('ingredient hidden', hidden[0].shape, hidden[1].shape)

        # unsort the output
        _, original_idx = sorted_idx.sort(0, descending=False)

        # LSTM ==>
        # bi-directional
        unsorted_idx = original_idx.view(1, -1, 1).expand_as(hidden[0])
        # 2 directions x batch_size x num features, we transpose 1st and 2nd dimension

        output = hidden[0].gather(1, unsorted_idx).transpose(0, 1).contiguous()
        output = output.view(output.size(0), output.size(1) * output.size(2))

        return output

# Im2recipe model
class ingre2recipe(nn.Module):
    def __init__(self):
        super(ingre2recipe, self).__init__()

        self.skip_embedding = nn.Sequential(
            nn.Linear(opts.srnnDim, opts.embDim),
            nn.Tanh(),
        )

        self.ingre_embedding = nn.Sequential(
            nn.Linear(opts.irnnDim * 2, opts.embDim),
            # nn.Tanh(),
            nn.ReLU(),
        )

        self.stRNN_ = stRNN()
        self.ingRNN_ = ingRNN()

        if opts.semantic_reg:
            self.semantic_branch = nn.Linear(opts.embDim, opts.numClasses)

    def forward(self, y1, y2, z1, z2):
        skip_emb = self.skip_embedding(self.stRNN_(y1, y2))
        skip_emb = norm(skip_emb)

        ingre_emb = self.ingre_embedding(self.ingRNN_(z1, z2))
        # ingre_emb = self.ingRNN_(z1, z2)
        ingre_emb = norm(ingre_emb)

        if opts.semantic_reg:
            ingre_sem = self.semantic_branch(ingre_emb)
            # final output
            output = [skip_emb, ingre_emb, ingre_sem]
        else:
            # final output
            output = [skip_emb, ingre_emb]
        return output

# Set transformer for ingredient representation
class SetTransformer(nn.Module):
    def __init__(self):
        """
        Arguments:
            in_dimension: an integer.  # 2
            out_dimension: an integer. # 5 * K
        """
        super(SetTransformer, self).__init__()
        in_dimension = opts.ingrW2VDim
        out_dimension = opts.embDim

        # _, vec = torchwordemb.load_word2vec_bin(opts.ingrW2V)
        w2v_model = gensim.models.KeyedVectors.load_word2vec_format(opts.ingrW2V, binary=True)
        vec = torch.FloatTensor(w2v_model.vectors)

        self.embs = nn.Embedding(vec.size(0), opts.ingrW2VDim, padding_idx=0)
        # padding: </s>, first one in word2vec dict

        self.embs.weight.data.copy_(vec)

        d = in_dimension
        m = 16  # number of inducing points
        h = 4  # number of heads
        k = 2  # number of seed vectors ???

        self.encoder = nn.Sequential(
            InducedSetAttentionBlock(d, m, h, RFF(d), RFF(d)),
            InducedSetAttentionBlock(d, m, h, RFF(d), RFF(d))
        )
        self.decoder = nn.Sequential(
            PoolingMultiheadAttention(d, k, h, RFF(d)),
            SetAttentionBlock(d, h, RFF(d))
        )

        self.decoder_2 = nn.Sequential(
            PoolingMultiheadAttention(d, k, h, RFF(d))
        )
        self.decoder_3 = nn.Sequential(
            SetAttentionBlock(d, h, RFF(d))
        )

        self.predictor = nn.Linear(k * d, out_dimension)

        # def weights_init(m):
        #     if isinstance(m, nn.Linear):
        #         nn.init.kaiming_normal_(m.weight)
        #         nn.init.zeros_(m.bias)
        #     elif isinstance(m, nn.LayerNorm):
        #         nn.init.ones_(m.weight)
        #         nn.init.zeros_(m.bias)
        #
        # self.apply(weights_init)

    def forward(self, x, sq_lengths):
        """
        Arguments:
            x: a float tensor with shape [batch, n, in_dimension].
        Returns:
            a float tensor with shape [batch, out_dimension].
        """
        x = self.embs(x) # shape [b, n, d]

        # print(x, x[1, 1, :].sum(), x[1, 2, :].sum(), x[3, 1, :].sum())
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        # exit()

        # sort sequence according to the length
        sorted_len, sorted_idx = sq_lengths.sort(0, descending=True)
        # batch_max_len = sorted_len.cpu().numpy()[0]
        batch_min_len = sorted_len.cpu().numpy()[0]

        cut_x = x[:, :batch_min_len, :]
        # print(cut_x, cut_x.shape)
        # output = cos(cut_x[1, :, :], cut_x[99, :, :])
        # print('embed ingre cutted', output.cpu().data)

        x = self.encoder(cut_x)  # shape [batch, batch_max_len, d]
        x = self.decoder(x)  # shape [batch, k, d]

        b, k, d = x.shape
        x = x.view(b, k * d)
        # print(x.shape)
        # output = cos(x[:8, :], x[70:78, :])
        # print('decoded viewed ingre cutted', output.cpu().data)

        y = self.predictor(x)
        # output = cos(y[:8, :], y[70:78, :])
        # print('y', output.cpu().data)
        # exit()
        return y

class Reciptor(nn.Module):
    def __init__(self):
        super(Reciptor, self).__init__()

        self.skip_embedding = nn.Sequential(
            nn.Linear(opts.srnnDim, opts.embDim),
            nn.Tanh(),
        )

        self.ingre_embedding = nn.Sequential(
            nn.Linear(opts.embDim, opts.embDim),
            nn.Tanh(),
        )

        self.stRNN_ = stRNN()
        self.ingSetTransformer_ = SetTransformer()

    def forward(self, y1, y2, z1, z2):
        skip_emb = self.skip_embedding(self.stRNN_(y1, y2))
        skip_emb = norm(skip_emb)

        ingre_emb = self.ingre_embedding(self.ingSetTransformer_(z1, z2))
        # ingre_emb = self.ingSetTransformer_(z1, z2)
        ingre_emb = norm(ingre_emb)

        output = [skip_emb, ingre_emb]
        return output

class CategoryClassification(nn.Module):
    def __init__(self):
        super(CategoryClassification, self).__init__()
        self.dropout = nn.Dropout(opts.hidden_dropout_prob)
        self.semantic_branch = nn.Linear(opts.embDim, opts.numClasses)

    def forward(self, ingre_emb):
        pooled_output = self.dropout(ingre_emb)
        recipe_class = self.semantic_branch(pooled_output)
        return recipe_class