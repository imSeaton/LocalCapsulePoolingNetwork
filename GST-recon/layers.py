from torch.nn import Parameter
import torch
from torch import nn
from set_transformer import SAB, PMA
from torch_geometric.utils import to_dense_batch

class P(nn.Module):
    def __init__(self, dim_input, dim_output, dim_hidden=128, num_heads=4, num_keys=[4, 1], ln=False, args=None):
        super(P, self).__init__()

        self.args = args

        self.pma = PMA(dim_input, num_heads, num_keys, ln=ln, mab=self.args.mab)

        self.lin = nn.Linear(dim_hidden, dim_output)

    def forward(self, X, batch, edge_index=None):

        batch_X, mask = to_dense_batch(X, batch)

        if len(batch_X.size()) == 4:
            batch_X = batch_X.squeeze(1)
        # attn is attentinon 
        X, attn = self.pma(batch_X, None, graph=(X, edge_index, batch), return_attention=True)
        X = self.lin(X)
        return X.squeeze(1), attn

class PSP(nn.Module):
    def __init__(self, dim_input, dim_output, dim_hidden=128, num_heads=4, num_keys=[4, 1], ln=False, args=None):
        super(PSP, self).__init__()

        self.args = args

        self.pma1 = PMA(dim_input, num_heads, num_keys[0], ln=ln, mab=self.args.mab)

        self.sab = SAB(dim_input, dim_hidden, num_heads, ln=ln, mab='MAB')
        
        self.pma2 = PMA(dim_hidden, num_heads, num_keys[1], ln=ln, mab='MAB')

        self.lin = nn.Linear(dim_hidden, dim_output)

    def forward(self, X, batch, edge_index=None):

        batch_X, mask = to_dense_batch(X, batch)
        
        extended_attention_mask = mask.unsqueeze(1)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -1e9

        X = self.pma1(batch_X, attention_mask=extended_attention_mask, graph=(X, edge_index, batch))
        X = self.sab(X)
        X = self.pma2(X)
        X = self.lin(X)
        return X.squeeze(1)

class PSSP(nn.Module):
    def __init__(self, dim_input, dim_output, dim_hidden=128, num_heads=4, num_keys=[4, 1], ln=False, args=None):
        super(PSSP, self).__init__()

        self.args = args

        self.pma1 = PMA(dim_input, num_heads, num_keys[0], ln=ln, mab=self.args.mab)

        self.sab1 = SAB(dim_input, dim_hidden, num_heads, ln=ln, mab='MAB')
        self.sab2 = SAB(dim_hidden, dim_hidden, num_heads, ln=ln, mab='MAB')

        self.pma2 = PMA(dim_hidden, num_heads, num_keys[1], ln=ln, mab='MAB')

        self.lin = nn.Linear(dim_hidden, dim_output)

    def forward(self, X, batch, edge_index=None):

        batch_X, mask = to_dense_batch(X, batch)
        
        extended_attention_mask = mask.unsqueeze(1)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -1e9

        X = self.pma1(batch_X, attention_mask=extended_attention_mask, graph=(X, edge_index, batch))
        X = self.sab1(X)
        X = self.sab2(X)
        X = self.pma2(X)
        X = self.lin(X)
        return X.squeeze(1)