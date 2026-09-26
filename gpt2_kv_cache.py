
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tiktoken
from torch.utils.data import Dataset, DataLoader


#Data Processing Code Components.

class GPTDataset_V1(Dataset):
    #For a class that inherits from abstract class o Dataset we need to incorporate __init_(), __len__() & __getitem__() methods.
    def __init__(self, text_corpus, tokenizer, max_length, stride=1):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []

        token_ids = self.tokenizer.encode(text_corpus)

        #Creating the dataset.
        for i in range(0, len(token_ids) - max_length, stride):  #Here we will by default use 1 stride for our text corpus.
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return  len(self.input_ids)

    def __getitem__(self, idx):
        return  self.input_ids[idx], self.target_ids[idx]



def create_dataloader_v1(text, max_length, stride, batch_size, shuffle):

    gpt_tokenizer = tiktoken.get_encoding('gpt2')
    dataset = GPTDataset_V1(text_corpus=text, tokenizer=gpt_tokenizer, max_length=max_length, stride=stride)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)

    return dataloader





#GELU class.
class GELU(nn.Module):

    def __init__(self):
        super().__init__()
    def forward(self, x):
        gelu_x = 0.5 * x * (1+ torch.tanh(
            torch.sqrt(torch.tensor(2.0/torch.pi)) * 
            (x+ 0.044715 * torch.pow(x,3))
            ))
        return gelu_x


#FFN class implementation.
class FeedForward(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        #Here now we creat a Two Linear Layers where in between we put a GELU ACtivation.
        #In Linear layers we want to project to higher dimension to learn the complex representation apply ACtivation and then bring it down to lower dimension.
        #Also We will use Sequential Here as we directly use th layers stack instead of tweaking in between results.
        self.layers = nn.Sequential(
            nn.Linear(in_features=cfg['emb_dim'], out_features= 4*cfg['emb_dim']),  #Projecting the embed_dim to 4 times 
            GELU(),
            nn.Linear(in_features=4*cfg['emb_dim'], out_features=cfg['emb_dim'])    #Then bringing it down to embed_dim
        )
    def forward(self, x):
        return self.layers(x)

    


#LayerNorm Class.
class LayerNorm(nn.Module):

    """In LayerNorm we don\'t just normalize the laer but also add a trainable Scale(sigma) & SHift(mu) learnable Parameters.
        Reason:
        1. We initialze the parameters with same size as embedding_dim_size with 1's for Scale Parameter
        2. ANd for Shift we initialize with 0's and then let the network learn them as needed.
        Why: Because for training stability we can keep the information normalized but to restrict it to not learnable would be too restrictive.
        That is why the papers use learable Scale(lambda) & Shift(beta) Parameters.
        Final Form: If X is the Normalized Data then Output: lambda * X + beta"""

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        #Nowe create the Parameters of size (embed_dim,) so we can use them as learnable paramtere and also boradcast with the whole input data.
        #But the scala & shift parameters work only on the embed_dim.
        self.scale = nn.Parameter(data=torch.ones(embed_dim))  #shape:(embed_dim,) so when multiplied elementwise boradcasted to (batch, num_token, embed_dim)
        self.shift = nn.Parameter(data=torch.zeros(embed_dim))

    def forward(self, x):

        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False) #We can have n-1 if unbiased=True for var but for huge embed dims n-1 or n doesn't matter as 

        norm_x = (x - mean) / torch.sqrt(var)

        #lambda * norm_x + beta = normalized(x)
        return  self.scale * norm_x + self.shift   #Here we are allowing the network to learn the scale & shift parameters if needed.





class MultiHeadAttentionCached(nn.Module):
    '''Class Implementation of KV Cache in MHA.'''

    def __init__(self, d_in, d_out, context_length, num_heads, dropout=0.1, qkv_bias=False, kv_cache=False):
        super().__init__()

        assert d_out % num_heads == 0, 'd_out must be divisible by num_heads'
        self.head_dim = d_out // num_heads
        self.num_heads = num_heads
        self.d_in, self.d_out, self.context_length = d_in, d_out, context_length
        self.kv_cache = kv_cache

        self.W_q = nn.Linear(in_features=d_in, out_features=d_out, bias=qkv_bias)
        self.W_k = nn.Linear(in_features=d_in, out_features=d_out, bias=qkv_bias)
        self.W_v = nn.Linear(in_features=d_in, out_features=d_out, bias=qkv_bias)

        self.register_buffer(
            name='mask',
            tensor=torch.triu(input=torch.ones(size=(context_length, context_length)), diagonal=1),
            persistent=False
        )

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(in_features=d_out, out_features=d_out)

        #Newly added components for KV Cache.
        self.register_buffer(
                    name='k_cache',
                    tensor=None,
                    persistent=False
                )
        self.register_buffer(
                            name='v_cache',
                            tensor=None,
                            persistent=False
                        )
        #Added to keep track of which position token for we are decoding / generating.
        self.ptr_current_pos = 0


    def forward(self, x):

        #x or input is of shape (batch_size, seq_length, d_in)
        #Note: If kv_cache=True then num_tokens is num_new_tokens.
        batch_size, num_tokens, d_in = x.shape

        #If kv_cache=True then Q, K_new, V_new size is (batch_size, new_seq_length, d_out)
        Q, K_new, V_new = self.W_q(x), self.W_k(x), self.W_v(x)  # create (batch_size, seq_length, d_in) ->(batch_size, seq_length, d_out)
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        Q = Q.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        K_new = K_new.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        V_new = V_new.view(batch_size, num_tokens, self.num_heads, self.head_dim)


        ## Newly added KV Cache Part.
        #If we use KV cache we can only pass new tokens and the older tokens will be stored in KV cache format.
        #As we are passing only new tokens so only new queries will be there.
        if self.kv_cache:
            # pass
            if self.k_cache == None:
                self.k_cache, self.v_cache = K_new, V_new
            else:
                #(batch_size, newly_added_seq_length, d_out) --> (batch_size, kv_cached_seq_length + newly_added_seq_length, d_out)
                self.k_cache, self.v_cache = torch.cat([self.k_cache, K_new], dim=1), torch.cat([self.v_cache, V_new], dim=1) #Concating along the num_tokens_dim
            #Once we have got the old Keys & Values concatenated withnew Ones now time for Attention Calculation.
            K, V = self.k_cache, self.v_cache
        else:
            #If no KV cache then we need to pass all tokens everytime to the model .
            K, V = K_new, V_new


        #Now we need to change calculate Attention for each block separately so first we have to interchange the num_tokens & num_heads place.
        Q, K, V = Q.transpose(1,2), K.transpose(1,2), V.transpose(1,2)  #Shape Output: (batch_size, num_heads,  num_tokens, head_dim)

        #If kv_cache == False : #Output Shape: (batch_size, num_heads,  seq_length, seq_length)
        #If kv_cache == True : #Output Shape: (batch_size, num_heads,  Q_num_tokens, K_num_tokens), so if we generate 1 token and pass 1 token then #Output Shape: (batch_size, num_heads,  1, K_num_tokens)
        #As we may generate 1 token at a time with only one newly added token each time, then we have to attend to each and every previous token than this new one.
        attn_scores = Q @ K.transpose(2,3)  


        ##Newly modified part in Causal Masking for KV cache .
        #Older Masking across whole Tokens.
        # attn_scores.masked_fill_(
        #     self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        # )
        #Here we are using the track keeping flag to do masking from 2th position to 5th position if we are decoding from 2 th position to 5th position.
        Q_num_tokens, K_num_tokens = Q.shape[-2], K.shape[-2] #Q/K shape:(batch_size, num_heads,  Q/K_num_tokens, head_dim)

        if self.kv_cache :
            mask_bool = self.mask.bool()[self.ptr_current_pos:self.ptr_current_pos + Q_num_tokens, : K_num_tokens]
            #Keeping track of last updated postions of the tokens.
            self.ptr_current_pos += Q_num_tokens
        else:
            mask_bool = self.mask.bool()[:Q_num_tokens, :K_num_tokens]


        attn_scores.masked_fill_(mask_bool, -torch.inf)


        attn_weights = torch.softmax(attn_scores/(K.shape[-1]**0.5), dim=-1)  #(batch_size, num_heads,  num_tokens , num_tokens)
        attn_weights = self.dropout(attn_weights)

        context_vectors = attn_weights @ V  #Output Shape: (batch_size, num_heads,  num_tokens, head_dim)
        #Reversing the position of num_heads & num_tokens
        context_vectors =  context_vectors.transpose(1,2)  #Output Shape: (batch_size, num_tokens, num_heads, head_dim)

        #Now we need to make the current context_vector shape memory contiguus and then couple the num_heads & head_dim into one.

        context_vectors = context_vectors.contiguous().view(batch_size, num_tokens, self.d_out)  #Merged the num_heads & head_dim into d_out.

        context_vectors = self.out_proj(context_vectors)
        return context_vectors

    def reset_cache(self):
        '''Function to reset the KV Cache Storage.'''
        self.k_cache, self.v_cache = None, None
        self.ptr_current_pos = 0






