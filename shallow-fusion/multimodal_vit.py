import torch
import torch.nn as nn
from transformers import RobertaModel
import torchvision.models as models
from collections import OrderedDict
import timm



class BertImage(nn.Module):
  def __init__(self, nfinetune=0):
    # Prep image transformer
    super(BertImage, self).__init__()
    vit_model =  timm.create_model('vit_base_patch16_224', pretrained=True)
    #self.vit_model =  torch.nn.Sequential(OrderedDict([*(list(vit_model.named_children())[:-1])]))
    self.vit_model = nn.Sequential(*list(vit_model.children())[:-1])
    for param in self.vit_model.parameters():
      param.requires_grad = True

    # Prep text transformer
    self.bert = RobertaModel.from_pretrained("FacebookAI/roberta-base")
    nhid_trans = self.bert.config.hidden_size

    for param in self.bert.parameters():
      param.requires_grad = False
    n_layers = 12
    if nfinetune > 0:
      for param in self.bert.pooler.parameters():
        param.requires_grad = True
      for i in range(n_layers-1, n_layers-1-nfinetune, -1):
        for param in self.bert.encoder.layer[i].parameters():
          param.requires_grad = True

    # classification layers
    self.binary_class = nn.Linear(nhid_trans*2, 1)
    self.sigmo = nn.Sigmoid()
    self.drop = nn.Dropout(0.1)

  def forward(self, spectos, input_ids, attention_mask):
      # encode the spectrogram
      vit_out = self.vit_model(spectos)
      img_embeddings = vit_out[:, 0, :]

      # encode transcription
      trans_embeddings = self.bert(input_ids, attention_mask=attention_mask)[0][:, 0]

      # concat two embeddings for shallow fusion
      embeddings = torch.concat((trans_embeddings, img_embeddings), dim=1)

      # run the final embedding through classification layers
      class_label = self.binary_class(embeddings)
      class_label = self.sigmo(class_label)
      return class_label

