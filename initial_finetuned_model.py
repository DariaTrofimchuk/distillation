import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from pytorch_lightning import LightningModule
from transformers import AutoModel, AdamW, get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, RandomSampler
from tqdm.notebook import tqdm


class Embedder(nn.Module):
    def __init__(self, model_path, freeze_bert, layer_num):
        super().__init__()

        self.model = AutoModel.from_pretrained(model_path)
        self.model.trainable = not freeze_bert
        self.bert_dim = self.model.config.hidden_size
        self.layer_num = layer_num
    
    def forward(self, input_ids, attention_mask):
        output = self.model(
           input_ids,
           attention_mask=attention_mask,
           return_dict=True,
           output_hidden_states=True
        )
        layer_embeddings = output.hidden_states[self.layer_num]
        embeddings = self.aggregate(layer_embeddings, attention_mask)
        norm = embeddings.norm(p=2, dim=1, keepdim=True)
        embeddings = embeddings.div(norm)
        return embeddings
    
    def aggregate(self, layer_embeddings, mask):
        raise NotImplementedError()


class AttentionEmbedder(Embedder):
    def __init__(self, model_path, freeze_bert, layer_num, hidden_dim, num_heads):
        super().__init__(model_path, freeze_bert, layer_num)

        self.num_heads = num_heads
        self.softmax = nn.Softmax(dim=1)
        
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.bert_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            for i in range(num_heads)
        ])
        
    def aggregate(self, layer_embeddings, mask):
        batch_size = layer_embeddings.size(0)
        num_tokens = layer_embeddings.size(1)
        hidden_dim = layer_embeddings.size(2)
        final_shape = (batch_size, hidden_dim * self.num_heads)
        final_embeddings = torch.zeros(final_shape, device=layer_embeddings.device)
        for head_num in range(self.num_heads):
            weights = self.softmax(self.heads[head_num](layer_embeddings).squeeze(-1))
            embeddings = weights.unsqueeze(1).bmm(layer_embeddings).squeeze(1)
            final_embeddings[:, head_num * hidden_dim:(head_num+1) * hidden_dim] = embeddings
        return final_embeddings
        

class MeanEmbedder(Embedder):
    def __init__(self, model_path, freeze_bert, layer_num, hidden_dim, use_masking=True):
        super().__init__(model_path, freeze_bert, layer_num)

        self.token_mapping = nn.Linear(self.bert_dim, hidden_dim)
        self.use_masking = use_masking

    def aggregate(self, layer_embeddings, mask):
        embeddings = self.token_mapping(layer_embeddings)
        if self.use_masking:
            expanded_mask = mask.unsqueeze(-1).expand(embeddings.size()).float()
            sum_embeddings = torch.sum(embeddings * expanded_mask, 1)
            sum_mask = torch.clamp(expanded_mask.sum(1), min=1e-9)
            return sum_embeddings / sum_mask
        return torch.mean(embeddings, dim=1)


class ClusteringTripletModel(LightningModule):
    def __init__(self, model_path, num_training_steps,
                 hidden_dim=256, freeze_bert=False,
                 layer_num=-1, margin=0.5, lr=1e-5):
        super().__init__()

        self.embedder = MeanEmbedder(
            model_path,
            freeze_bert=freeze_bert,
            layer_num=layer_num,
            hidden_dim=hidden_dim
        )

        self.triplet_loss = nn.TripletMarginWithDistanceLoss(
            margin=margin,
            distance_function=nn.PairwiseDistance(p=2)
        )

        self.lr = lr
        self.num_training_steps = num_training_steps

    def forward(self, pivots, positives, negatives):
        pivot_embeddings = self.embedder(pivots["input_ids"], pivots["attention_mask"])
        positive_embeddings = self.embedder(positives["input_ids"], positives["attention_mask"])
        negative_embeddings = self.embedder(negatives["input_ids"], negatives["attention_mask"])
        loss = self.triplet_loss(pivot_embeddings, positive_embeddings, negative_embeddings)
        return loss

    def training_step(self, batch, batch_nb):
        train_loss = self(*batch)
        return train_loss

    def validation_step(self, batch, batch_nb):
        val_loss = self(*batch)
        self.log("val_loss", val_loss, prog_bar=True, logger=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        return [optimizer]