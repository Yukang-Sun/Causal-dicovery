import math

import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super(CausalSelfAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.scaling = (d_model // nhead) ** -0.5
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)
        self.output = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, d_model = x.size()

        # Linear projections
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.nhead, d_model // self.nhead).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.nhead, d_model // self.nhead).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.nhead, d_model // self.nhead).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scaling

        # Create causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

        # Apply the mask
        attn_scores.masked_fill_(causal_mask, float("-inf"))

        # Softmax to get attention weights
        attn_weights = self.softmax(attn_scores)

        # Compute attention output
        attn_output = torch.matmul(attn_weights, V)

        # Reshape back to original shape
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        # Final linear projection
        output = self.output(attn_output)

        return output, attn_weights  # Return both output and attention weights


class DilatedConvBlock(nn.Module):
    def __init__(self, d_model, dilation):
        super(DilatedConvBlock, self).__init__()
        self.dilation = dilation
        self.conv = nn.Conv1d(
            d_model, d_model, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.norm = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x.transpose(1, 2)  # Change to (batch_size, d_model, seq_len) for Conv1d
        x = self.conv(x)
        x = x.transpose(1, 2)  # Change back to (batch_size, seq_len, d_model)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dilation, dropout=0.1):
        super(TransformerBlock, self).__init__()
        # Dilated convolution block before self-attention
        self.conv_block = DilatedConvBlock(d_model, dilation)
        self.self_attn = CausalSelfAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        # Apply dilated convolution before self-attention
        src2 = self.conv_block(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Self-attention block
        src2, attention_weights = self.self_attn(src)  # Get attention weights from CausalSelfAttention
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        # Feedforward network
        src2 = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src, attention_weights  # Return transformed output and attention weights


class Transformer(nn.Module):
    def __init__(
        self,
        num_layers,
        d_model,
        nhead,
        dim_feedforward,
        input_dim,
        output_dim,
        dilation,
        dropout=0.1,
    ):
        super(Transformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, nhead, dim_feedforward, dilation, dropout)
                for _ in range(num_layers)
            ]
        )
        self.output_layer = nn.Linear(d_model, output_dim)

    def forward(self, src):
        # Embedding and positional encoding
        src = self.embedding(src)
        src = self.pos_encoder(src)

        attention_weights = None  # Initialize attention weights variable

        # Iterate through layers
        for i, layer in enumerate(self.layers):
            src, attention_weights = layer(src)  # Only attention weights of the last layer will be retained

        # Final output layer
        predicted_output = self.output_layer(src)
        return predicted_output, attention_weights  # Return both predicted output and attention weights of the last layer


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=30000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)