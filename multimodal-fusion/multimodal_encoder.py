import torch
import torch.nn as nn
from transformers import RobertaModel
import torchvision.models as models
from collections import OrderedDict
import timm

class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.2):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class CrossAttention(nn.Module):
  def __init__(self, input_dim):
    super(CrossAttention, self).__init__()
    self.input_dim = input_dim
    self.query = nn.Linear(input_dim, input_dim) # [batch_size, seq_length, input_dim]
    self.key = nn.Linear(input_dim, input_dim) # [batch_size, seq_length, input_dim]
    self.value = nn.Linear(input_dim, input_dim)
    self.softmax = nn.Softmax(dim=2)

  def forward(self, Q, K): # x.shape (batch_size, seq_length, input_dim)
    queries = self.query(Q)
    keys = self.key(K)
    values = self.value(K)

    scores = torch.bmm(queries, keys.transpose(1, 2))/(self.input_dim**0.5)
    attention = self.softmax(scores)
    weighted = torch.bmm(attention, values)
    return weighted



class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        # '-1' means last dimension.

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out


class EncoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden=768, drop_prob=0.6):
        super(EncoderLayer, self).__init__()
        self.cross_attention = CrossAttention(d_model)
        self.norm1 = LayerNorm(dim=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(dim=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, query, key):
        # compute self attention
        _x = query
        x = self.cross_attention(query, key)
        
        # add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        
        # positionwise feed forward network
        _x = x
        x = self.ffn(x)
      
        # add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x



class BertImage(nn.Module):
  def __init__(self, nfinetune=0):
    super(BertImage, self).__init__()
    vit_model =  timm.create_model('vit_base_patch16_224', pretrained=True)
    self.vit_model =  torch.nn.Sequential(OrderedDict([*(list(vit_model.named_children())[:-1])]))
    for param in self.vit_model.parameters():
      param.requires_grad = True


    self.bert = RobertaModel.from_pretrained("FacebookAI/roberta-base")
    nhid = self.bert.config.hidden_size

    for param in self.bert.parameters():
      param.requires_grad = False
    n_layers = 12
    if nfinetune > 0:
      for param in self.bert.pooler.parameters():
        param.requires_grad = True
      for i in range(n_layers-1, n_layers-1-nfinetune, -1):
        for param in self.bert.encoder.layer[i].parameters():
          param.requires_grad = True


    self.cme_img = EncoderLayer(nhid)
    self.cme_trans = EncoderLayer(nhid)

    self.pooler = nn.AvgPool1d(5, stride=3)


    self.binary_class = nn.Linear(nhid*2, 1)
    self.sigmo = nn.Sigmoid()
    self.drop = nn.Dropout(0.1)

  def forward(self, spectos, input_ids, attention_mask):
      img_embeddings = self.vit_model(spectos)

      trans_embeddings = self.bert(input_ids, attention_mask=attention_mask)[0]

      crossed_img = self.cme_img(img_embeddings, trans_embeddings)
      crossed_trans = self.cme_trans(trans_embeddings, img_embeddings)


      crossed_img = torch.mean(crossed_img, dim=1)
      crossed_trans = torch.mean(crossed_trans, dim=1)

      embeddings = torch.concat((crossed_img, crossed_trans), dim=1)
      embeddings = self.pooler(embeddings)

      class_label = self.binary_class(embeddings)
      class_label = self.sigmo(class_label)
      return class_label

