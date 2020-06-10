import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.nn import CrossEntropyLoss, MSELoss
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
# import torchwordemb
from args import get_parser
from model.blocks import *
import gensim
import os
# ref: https://github.com/huggingface/transformers/blob/74ce8de7d8e0375a9123f9542f3483f46cc8df9b/examples/run_glue.py

# ================================================================================
parser = get_parser()
opts = parser.parse_args()
# # ==============================================================================

class TableModule(nn.Module):
    def __init__(self):
        super(TableModule, self).__init__()

    def forward(self, x, dim):
        y = torch.cat(x, dim)
        return y

# normalization
def norm(input, p=2, dim=1, eps=1e-12):
    return input / input.norm(p, dim, keepdim=True).clamp(min=eps).expand_as(input)

# Skip-thoughts LSTM for handling instructions
class stRNN(nn.Module):
    def __init__(self):
        super(stRNN, self).__init__()
        # keep the same embedding
        self.lstm = nn.LSTM(input_size=opts.stDim, hidden_size=opts.srnnDim, bidirectional=False, batch_first=True)

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

        # print(sorted_inputs, sorted_inputs.shape, sorted_len)
        # sorted_inputs.shape != sorted_len

        # pass it to the lstm, didn't use hidden here
        out, hidden = self.lstm(packed_seq)
        # print('st out\n', out)

        # unsort the output
        _, original_idx = sorted_idx.sort(0, descending=False)

        unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        unsorted_idx = original_idx.view(-1, 1, 1).expand_as(unpacked)
        # print(unpacked.shape)

        # we get the last index of each sequence in the batch
        idx = (sq_lengths - 1).view(-1, 1).expand(unpacked.size(0), unpacked.size(2)).unsqueeze(1)
        # print(idx)

        # we sort and get the last element of each sequence
        # [n steps, lstm the get the last step have the former n-1 steps info, thus only get the last one]
        output = unpacked.gather(0, unsorted_idx.long()).gather(1, idx.long())
        # print('out put', output, output.shape)

        output = output.view(output.size(0), output.size(1) * output.size(2))

        # print('skip lstm output', output.shape)
        # exit()
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

        # print('sorted_inputs, sorted_inputs.shape, sorted_len', sorted_inputs.shape, sorted_len)
        # exit()

        # pass it to the rnn
        # output(seq_len, batch, hidden_size * num_directions)
        # h_n(num_layers * num_directions, batch, hidden_size)

        out, hidden = self.irnn(packed_seq)
        # print('ingredient hidden', hidden[0].shape, hidden[1].shape)

        # unsort the output
        _, original_idx = sorted_idx.sort(0, descending=False)

        # LSTM ==> ???? why use hidden[0] ???
        # bi-directional
        unsorted_idx = original_idx.view(1, -1, 1).expand_as(hidden[0])
        # 2 directions x batch_size x num features, we transpose 1st and 2nd dimension
        # torch.Size([2, 3, 300])
        output = hidden[0].gather(1, unsorted_idx).transpose(0, 1).contiguous()
        # print(output.shape)
        output = output.view(output.size(0), output.size(1) * output.size(2))
        # print(output.shape)

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
            nn.Tanh(),
        )

        self.recipe_embedding = nn.Sequential(
                nn.Linear(opts.irnnDim * 2 + opts.srnnDim, opts.embDim),
                nn.Tanh(),
            )

        self.stRNN_ = stRNN()
        self.ingRNN_ = ingRNN()
        self.table = TableModule()

        if opts.semantic_reg:
            self.semantic_branch = nn.Linear(opts.embDim, opts.numClasses)

    def forward(self, y1, y2, z1, z2):  # we need to check how the input is going to be provided to the model
        # recipe embedding
        # recipe_emb = self.table([self.stRNN_(y1, y2), self.ingRNN_(z1, z2)], 1)  # joining on the last dim
        # # print(recipe_emb.shape)
        #
        # recipe_emb = self.recipe_embedding(recipe_emb)
        # recipe_emb = norm(recipe_emb)

        skip_emb = self.skip_embedding(self.stRNN_(y1, y2))
        skip_emb = norm(skip_emb)

        ingre_emb = self.ingre_embedding(self.ingRNN_(z1, z2))
        ingre_emb = norm(ingre_emb)

        # visual embedding
        # visual_emb = self.visionMLP(x)
        # visual_emb = visual_emb.view(visual_emb.size(0), -1)
        # visual_emb = self.visual_embedding(visual_emb)
        # visual_emb = norm(visual_emb)

        if opts.semantic_reg:
            # visual_sem = self.semantic_branch(visual_emb)
            # recipe_sem = self.semantic_branch(recipe_emb)

            skip_sem = self.semantic_branch(skip_emb)
            ingre_sem = self.semantic_branch(ingre_emb)

            # final output
            # output = [visual_emb, recipe_emb, visual_sem, recipe_sem]
            # output = [recipe_emb, recipe_sem]

            output = [skip_emb, ingre_emb, skip_sem, ingre_sem]
        else:
            # final output
            # output = [visual_emb, recipe_emb]
            # output = [recipe_emb]
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
        out_dimension = opts.irnnDim

        # self.irnn = nn.LSTM(input_size=opts.ingrW2VDim, hidden_size=opts.irnnDim, bidirectional=True, batch_first=True)

        # _, vec = torchwordemb.load_word2vec_bin(opts.ingrW2V)
        w2v_model = gensim.models.KeyedVectors.load_word2vec_format(opts.ingrW2V, binary=True)
        vec = torch.FloatTensor(w2v_model.vectors)

        self.embs = nn.Embedding(vec.size(0), opts.ingrW2VDim, padding_idx=0)  # not sure about the padding idx
        # padding: </s>, first one in word2vec dict

        self.embs.weight.data.copy_(vec)

        # d = 128
        d = in_dimension
        m = 16  # number of inducing points
        h = 4  # number of heads
        k = 4  # number of seed vectors ???

        self.encoder = nn.Sequential(
            InducedSetAttentionBlock(d, m, h, RFF(d), RFF(d)),
            InducedSetAttentionBlock(d, m, h, RFF(d), RFF(d))
        )
        self.decoder = nn.Sequential(
            PoolingMultiheadAttention(d, k, h, RFF(d)),
            SetAttentionBlock(d, h, RFF(d))
        )

        self.predictor = nn.Linear(k * d, out_dimension)

        def weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        self.apply(weights_init)

    def forward(self, x, sq_lengths):
        """
        Arguments:
            x: a float tensor with shape [batch, n, in_dimension].
        Returns:
            a float tensor with shape [batch, out_dimension].
        """
        # we get the w2v for each element of the ingredient sequence
        x = self.embs(x) # shape [b, n, d]

        # sort sequence according to the length
        sorted_len, sorted_idx = sq_lengths.sort(0, descending=True)
        batch_max_len = sorted_len.cpu().numpy()[0]
        # print(batch_max_len)
        cut_x = x[:, :batch_max_len, :]
        # print(cut_x, cut_x.shape)

        x = self.encoder(cut_x)  # shape [batch, batch_max_len, d]
        x = self.decoder(x)  # shape [b, k, d]

        b, k, d = x.shape
        x = x.view(b, k * d)
        return self.predictor(x)

class Reciptor(nn.Module):
    def __init__(self):
        super(Reciptor, self).__init__()

        self.skip_embedding = nn.Sequential(
            nn.Linear(opts.srnnDim, opts.embDim),
            nn.Tanh(),
        )

        self.ingre_embedding = nn.Sequential(
            nn.Linear(opts.irnnDim, opts.embDim),
            nn.Tanh(),
        )

        self.stRNN_ = stRNN()
        self.ingRNN_ = ingRNN()
        self.ingSetTransformer_ = SetTransformer()
        # self.table = TableModule()

        if opts.semantic_reg:
            self.semantic_branch = nn.Linear(opts.embDim, opts.numClasses)

    def forward(self, y1, y2, z1, z2):
        skip_emb = self.skip_embedding(self.stRNN_(y1, y2))
        skip_emb = norm(skip_emb)

        ingre_emb = self.ingre_embedding(self.ingSetTransformer_(z1, z2))
        ingre_emb = norm(ingre_emb)

        if opts.semantic_reg:
            ingre_sem = self.semantic_branch(ingre_emb)
            output = [skip_emb, ingre_emb, ingre_sem]
        else:
            # final output
            output = [skip_emb, ingre_emb]
        return output

# TODO: model for evaluation
class BertPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__()
        self.config = opts

    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        # elif isinstance(module, BertLayerNorm):
        #     module.bias.data.zero_()
        #     module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_path):
        """
        Instantiate a BertPreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.
        Params:
            pretrained_model_name_or_path: either:
                - a str with the name of a pre-trained model to load selected in the list of:
                    . `bert-base-uncased`
                    . `bert-large-uncased`
                    . `bert-base-cased`
                    . `bert-large-cased`
                    . `bert-base-multilingual-uncased`
                    . `bert-base-multilingual-cased`
                    . `bert-base-chinese`
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `model.chkpt` a TensorFlow checkpoint
            from_tf: should we load the weights from a locally saved TensorFlow checkpoint
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models

            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        # Instantiate model.
        parser = get_parser()
        opts = parser.parse_args()
        model = cls()
        print('test initial model', model)

        weights_path = os.path.join(pretrained_model_path)
        state_dict = torch.load(weights_path, map_location='cpu')
        print('state_dict', state_dict)

        # Load from a PyTorch state_dict

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        start_prefix = ''
        print(start_prefix)

        load(model, prefix=start_prefix)
        print(model)

        # model = TheModelClass(*args, **kwargs)
        # model.load_state_dict(torch.load(PATH))
        return model

# code from hugface transformer program:
class DStaskForSequenceClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]
    """
    def __init__(self):
        super(DStaskForSequenceClassification, self).__init__()
        self.num_labels = opts.num_labels

        self.bert = ingre2recipe()
        self.dropout = nn.Dropout(opts.hidden_dropout_prob)
        self.classifier = nn.Linear(opts.hidden_size, opts.num_labels)

        # self.init_weights()

    def forward(self, y1=None, y2=None, z1=None, z2=None, labels=None):

        # inputs = {'input_ids': batch[0],
        #           'attention_mask': batch[1],
        #           'labels': batch[3]}
        #
        # if args.model_type != 'distilbert':
        #     inputs['token_type_ids'] = batch[2] if args.model_type in ['bert',
        #                                                                'xlnet'] else None

        outputs = self.bert(y1, y2, z1, z2)
        # [skip_emb, ingre_emb, skip_sem, ingre_sem]
        pooled_output = outputs[0]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # print(logits, labels)
        # exit()
        # outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                # print(loss)
                # exit()
            outputs = (loss,) + tuple(outputs)

        return (loss, logits)  # (loss), logits, (hidden_states), (attentions)