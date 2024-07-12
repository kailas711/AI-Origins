import torch
import torch.nn as nn 
import math 

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(d_model, vocab_size)

    def forward(self, x):
        # as per the paper the embeddings are multiplied by sqrt(d_model)
        return self.embedding(x) * math.sqrt(self.d_model)
    
# We need to tell the model where each word belong in a setence. 
# This is done via positional encoding vector which is of size 512 (same as embeddings)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int , dropout: float):
        super().__init__()
        self.d_model = d_model,
        self.seq_len = seq_len,
        self.dropout = nn.Dropout(dropout)

        # Create the metrics [pe]. We'll use a slightly simplified version of the formula 
        # When you apply the exp() and the log() of something inside the exp() the result is 
        # a number that is more numericaly stable. 
        pe = torch.zeros(seq_len, d_model)

        #create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (math.log(10000.0) / d_model))
        
        # Apply sine to even positions 
        pe[: , 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd positions 
        pe[:, 1::2] = torch.cos(position * div_term)
        # Add a dimension for batch size
        pe = pe.unsqueeze(0) # (1, seq_len, dmodel)

        self.register_buffer('pe', pe)
        # The register_buffer saves the parameter in the model save file along with the state of the model

    def forward(self, x):
        # We don't want the model to learn PE , as for the sentence the positional encoding remains same 
        x = x + (self.pe[: ,x.shape[1], :]).requires_grad(False)
        return self.dropout(x)
    
class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10**-6):
        super().__init__()
        self.eps = eps # for numerical stability , prevets zero division error
        self.alpha = nn.Parameter(torch.ones(1)) # Gets Multiplied 
        self.bias = nn.Parameter(torch.zeros(1)) # gets Added 

    def forward(self, x):
        mean = x.mean(dim = -1, keepdims  = True) # keeps the og dimension
        std = x.std(dim = -1, keepdims = True)
        # Apply the formula 
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and B2

    def forward(self, x):
        # The input (batch, seq_len, d_model) is converted into ---> (batch, seq_len, d_ff) in linear_1 then 
        # after linear_2 it is converted to (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

# The multihead attention takes in (Q,K,V) with dim (seq, d_model) which is the 3 duplicate of inputs then we multiply those (Q,K,V) by the matrices 
# Wq Wk and Wv with dim (d_model, d_model) . These further splits into matrices as per number of heads

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float):
        super.__init__()
        self.d_model = d_model
        self.h = h 
        # We need to make sure that d_model is divisible by h , only then we can split it properly as mentioned above (these are the multi heads)
        assert d_model % h == 0, "d_model is not divisible by no: of heads (h)"
        
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # The output matrix 
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout1d(dropout)

    # staticmethod can be called anywhere without the instance of the class. 
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        # If don't want some words to be considered, replace mask with small value which get zeroed in softmax
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim= -1) # (batch, h, seq_len, seq_len)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores*value), attention_scores # this score alone will be used for visulizations


    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (Batch, seq_len, d_model)
        key = self.w_k(k) # (Batch, seq_len, d_model)
        value = self.w_v(v) # (Batch, seq_len, d_model)

        # We need to divide the query into multiple matrices as to match no: of heads 
        # we are going from (Batch, seq_len, d_model) ----> (Batch, seq_len, h, d_k) ---> (Batch, h ,seq_len, d_k) via transpose
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2) # transpose cuz we need the attention to see the 2nd dim first
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

        x, self.attention_score = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # transpose back to concat them the resultant is (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        
        # (batch, h, seq_len, d_k) ---> (batch)
        return self.w_o(x)
    
class ResidualConnection(nn.Module):

    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock,
                 dropout: float):
        super().__init__()    
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    # src_mask is to prevent the interaction of padding token with others
    def forward(self, x, src_mask):