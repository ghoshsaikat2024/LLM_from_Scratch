
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tiktoken
from torch.utils.data import Dataset, DataLoader


'''
This File has Two types of KV Cache Implementation 

1. 1st one is Naive / Simple KV Ccahing.
 Caveat : This implementation has one major Caveat which is if the input_sequence is equal to context_size of model or during KV Caching
    the context size is filled since we are already have the KV cached so this one doesn't work at all.
    For this problem we need to use a window size like rolling KV Cache Mechanism.

2. 2nd One is Rolling Window based KV Caching where we take the latest context size length for KV Caching of Previous Tokens.
Note: This one hasn't been implemented yet.

'''




#Test GPT2 Configuration.
#GPT2 Configuration.
GPT_CONFIG_124M = {
    "vocab_size": 50257, # Vocabulary size
    "context_length": 256, # Context length  #Change to 256 from 1024 for Learning
    "emb_dim": 768, # Embedding dimension
    "n_heads": 12, # Number of attention heads
    "n_layers": 12, # Number of layers i.e specifies the number of Transformer blocks in the model
    "drop_rate": 0.1, # Dropout rate
    "qkv_bias": False # Query-Key-Value bias
}



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



#Naive/Simple KV Cache Implementation in MultiHeadAttention.
class MultiHeadAttentionCachedSimple(nn.Module):
    '''Class Implementation of KV Cache in MHA.

    A. Caveat : This implementation has one major Caveat which is if the input_sequence is equal to context_size of model or during KV Caching
        the context size is filled since we are already have the KV cached so this one doesn't work at all.
        For this problem we need to use a window size like rolling KV Cache Mechanism.

    KV Cache Storing :

    Concretely, after the cache is initialized via the if self.cache_k is None: ..., we add 
    the newly generated keys and values via self.cache_k = torch.cat(...) and self.cache_v = torch.cat(...) to the cache, respectively.
    
    Retrieving

    Then, keys, values = self.cache_k, self.cache_v retrieves the stored values and keys from the cache.
    
    '''

    def __init__(self, d_in, d_out, context_length, num_heads, dropout=0.1, qkv_bias=False):
        super().__init__()

        assert d_out % num_heads == 0, 'd_out must be divisible by num_heads'
        self.head_dim = d_out // num_heads
        self.num_heads = num_heads
        self.d_in, self.d_out, self.context_length = d_in, d_out, context_length

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
        #Since KV Cache is used only in LLM generation/Serving/Inference Mode so we dont make it part of the model itself
        #that's why the tensor for it is not a persistent one.
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
        #This is a simple counter that remembers how many tokens the model has already cached during an incremental generation session.
        self.ptr_current_pos = 0


    def forward(self, x, kv_cache=False):  #KV Cache is used during the forward pass only.

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
        #If we use KV cache we can only pass new tokens and the older tokens will be stored in KV cache format as an atribute of the attn class.
        #As we are passing only new tokens so only new queries will be there.
        if kv_cache:
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
        attn_scores = Q @ K.transpose(2,3)  #Output Shape (batch_size, num_heads,  Q_num_tokens, K_num_tokens)


        ############ Newly modified part in Causal Masking for KV cache . ############
        #Older Masking across whole Tokens.
        # attn_scores.masked_fill_(
        #     self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        # )
        #Here we are using the track keeping flag to do masking from 2th position to 5th position if we are decoding from 2 th position to 5th position.
        Q_num_tokens, K_num_tokens = Q.shape[-2], K.shape[-2] #Q/K shape:(batch_size, num_heads,  Q/K_num_tokens, head_dim)

        if kv_cache :
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
        '''Function to reset the KV Cache Storage & Pointer to the Latest Tokens.
        
        When generating text, we have to remember to reset both the keys and value buffers between two separate text-generation calls. 
        Otherwise, the queries of a new prompt will attend to stale keys left over from the previous sequence, which causes the model to 
        rely on irrelevant context and produce incoherent output. To prevent this, we add a reset_kv_cache method to the MultiHeadAttention Block.
        '''
        
        self.k_cache, self.v_cache = None, None
        self.ptr_current_pos = 0



#Naive/Simple KV Cache Implementation on Transformer Block.
class TransformerBlockCachedSimple(nn.Module):

    def __init__(self, cfg):
        super().__int__()

        self.attn = MultiHeadAttentionCachedSimple(
            d_in = cfg['emb_dim'],
            d_out = cfg['emb_dim'],
            context_length = cfg['context_length'],
            num_heads = cfg['n_heads'],
            dropout = 0.1,
            qkv_bias = cfg['qkv_bias']
        )
        self.ffn = FeedForward(cfg=cfg)
        self.norm1 = LayerNorm(embed_dim=cfg['emb_dim'])
        self.norm2 = LayerNorm(embed_dim=cfg['emb_dim'])

        self.dropout = nn.Dropout(p=cfg['drop_rate'])


    def forward(self, x, kv_cache=False):

        # Shortcut connection for attention block
        shortcut = x #Skip Connection.

        x = self.norm1(x)
        #Newly modified kv cache line during forward pass
        x = self.attn(x, kv_cache=kv_cache)
        x = self.dropout(x)

        #Adding shortcut/skip connection.
        x = x+shortcut

        ## SHortcut connection for FFN block
        shortcut = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout(x)

        x = x + shortcut

        return  x



#Gpt Model Class KV Cache Implementation.
class GPTModelCachedSimple(nn.Module):
    '''KV Cache Implementation GPT Model Class.
    A. Caveat : This implementation has one major Caveat which is if the input_sequence is equal to context_size of model or during KV Caching
        the context size is filled since we are already have the KV cached so this one doesn't work at all.
        For this problem we need to use a window size like rolling KV Cache Mechanism.
    '''

    def __init__(self, cfg):
        super().__init__()

        self.token_emb = nn.Embedding(num_embeddings=cfg['vocab_size'], embedding_dim=cfg['emb_dim'])
        self.pos_emb = nn.Embedding(num_embeddings=cfg['context_length'], embedding_dim=cfg['emb_dim'])

        ## Newly Modified one after KV Cache.
        # self.trf_blocks = nn.Sequential(
        #             *[TransformerBlock(cfg) for _ in range(cfg['n_layers'])]
        #         )
        self.trf_blocks = nn.ModuleList(
            [TransformerBlockCachedSimple(cfg) for _ in range(cfg['n_layers'])]
        )

        #We also need a Position Counter here as well for tracking for many tokens have been cached.
        self.ptr_current_pos = 0

        self.final_norm = LayerNorm(embed_dim=cfg['emb_dim'])
        self.dropout = nn.Dropout(p=cfg['drop_rate'])
        self.out_head = nn.Linear(in_features=cfg['emb_dim'], out_features=cfg['vocab_size'], bias=False)

    def forward(self, in_idx, kv_cache=False):

        batch_size, seq_len = in_idx.shape

        token_emb = self.token_emb(in_idx)    #(batch_size,seq_length, embed_dim)

        #Newly Modified Part.
        if kv_cache:
            pos_ids = torch.arange(start=self.ptr_current_pos, end=self.ptr_current_pos+seq_len, device=in_idx.device, dtype=torch.long)

        else:
            pos_ids = torch.arange(start=0, end=seq_len, device=in_idx.device, dtype=torch.long) #Shape(seq_length,)

        #Shape(seq_length,embed_dim) -> (1, seq_length, embed_dim) SO now we can add this with the toke_emb using broadcasting which is of size :(batch_size,seq_length, embed_dim)
        pos_emb = self.pos_emb(pos_ids).unsqueeze(0)   #

        x = token_emb + pos_emb
        x = self.drop_emb(x)

        #Passing through n_layers of Transformer Blocks.
        for trf in self.trf_blocks:
            x = trf(x, kv_cache=kv_cache)

        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

    #Now we add one extra method like we did in Attention Class to reset KV cache across all Transformer Blocks.
    def reset_kv_cache(self):
        '''During Generation we call this method at starting of each sequence generation so for each sequence token generation we dont have to pass
             the whole previous sequence rather only the new token only.
            ANd also to make sure each sequence is independently generated we flush the KV Cache while processing each new sequence.'''
        for trf in self.trf_blocks:
            trf.attn.reset_cache()
        self.ptr_current_pos = 0




#Greedy Style Text Deoing with Simle KV Cache.
def generate_text_greedy_cached_simple(model, idx, context_length, max_new_tokens, kv_cache=True):

    '''KV Cache Implementation on Greedy Decoding.
    Note:

    A. Caveat : This implementation has one major Caveat which is if the input_sequence is equal to context_size of model or during KV Caching
    the context size is filled since we are already have the KV cached so this one doesn't work at all.
    For this problem we need to use a window size like rolling KV Cache Mechanism.
    '''

    model.eval()
    with torch.no_grad() :

        if kv_cache:
            # pass
            # Init cache with full prompt
            model.reset_kv_cache()
            #Input Shape:(batch, seq_len)
            idx = idx[:, -context_length:] 
            logits = model(idx, kv_cache=kv_cache)

            for _ in range(max_new_tokens):
                #  logits = logits[:, -1, :]  #One can also equivalently write : logits[:, -1] means the same thing
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)  #Another way to write things
                 # b) append it to the running sequence
                idx = torch.cat([idx, next_idx], dim=1)
                ## c) feed model only the new token
                logits = model(next_idx, kv_cache=True)



        else:
            for _ in range(max_new_tokens):
                #Input Shape:(batch, seq_len)
                idx = idx[:, -context_length:] 
                logits = model(idx, kv_cache=False)  ##Output Shape:(batch, seq_len, vocab_dim)
            
                logits = logits[:, -1, :]           ##Output Shape:(batch, 1, vocab_dim)
                #Using Greedy decoding for only the max logit value.
                idx_next = torch.argmax(input=logits, dim=-1, keepdim=True)   #Output Shape:(batch, 1)

                idx = torch.cat(tensors=[idx, idx_next], dim=-1) ## (batch, n_tokens+1)
        
        

    return  idx





