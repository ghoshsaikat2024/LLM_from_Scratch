import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tiktoken
from torch.utils.data import Dataset, DataLoader



'''
This Script contains MultiHead Latent Attention Implementation on GPT-2 Architecture along with Naive KV Cache.
MLA instead of storing the Key , value Vectors of Matrix for batches stores a generalized latent small dimensional intermediate vectors /Matrix for unique token/batch tokens.
Checkout this article for Resource on Multi Head Latent Attention: https://machinelearningmastery.com/a-gentle-introduction-to-multi-head-latent-attention-mla/
Also Check out this Repo by Sebastian Raschka : https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/05_mla
'''


#Test GPT2 Configuration.
#GPT2 Configuration.
GPT_CONFIG_124M = {
    "vocab_size": 50257, # Vocabulary size
    "context_length": 256, # Context length  #Change to 256 from 1024 for Learning
    "emb_dim": 768, # Embedding dimension
    "n_heads": 12, # Number of attention heads
    "n_layers": 12, # Number of layers i.e specifies the number of Transformer blocks in the model
    'num_kv_groups': 3, #Number of KV groups and we need to assign multiple heads to each one of these groups.
    "latent_dim" : 12, #For Multi HEad Latent Attention use latent_dim other wise for GQA use num_kv_groups.
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




class MultiHeadLatentAttentionCachedSimple(nn.Module):


    def __init__(self, d_in, d_out, context_length, num_heads, dropout, latent_dim, qkv_bias=False, dtype=None):
        super().__init__()

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        assert d_out % num_heads == 0, 'd_out must be divisible by num_heads'

        self.num_heads, self.head_dim = num_heads, d_out // num_heads
        self.latent_dim = latent_dim if latent_dim is not None else max(16,d_out // num_heads)
        self.d_in, self.d_out, self.context_length = d_in, d_out, context_length

        #Initializing the W_q matrix.
        self.W_q = nn.Linear(in_features=d_in, out_features=d_out, qkv_bias=qkv_bias, device=device, dtype=dtype)

        #Initializing the Latent Downsampling Matrix & Also Upsampling Matrix as well.
        self.W_kv = nn.Linear(in_features=d_in, out_features=latent_dim, qkv_bias=qkv_bias, device=device, dtype=dtype)
        self.W_ku = nn.Linear(in_features=latent_dim, out_features=d_out, qkv_bias=qkv_bias, device=device, dtype=dtype)
        self.W_vu = nn.Linear(in_features=latent_dim, out_features=d_out, qkv_bias=qkv_bias, device=device, dtype=dtype)

        #Tracking Pointer to keep track of Cached latent C_kv vector .
        self.ptr_current_pos = 0

        self.dropout = nn.Dropout(p=dropout)

        self.out_proj = nn.Linear(in_features=d_out, out_features=d_out, qkv_bias=None, device=device, dtype=dtype)

        #Creating the KV Cache tensor.
        self.register_buffer(
            name='c_kv_cache',
            tensor=None,
            persistent=False    #Since KV Cache is not part of Model so persistent memory is False.
        )



    def forward(self, x, kv_cache=False):

        batch_size, num_tokens, d_in = x.shape  #Shape:(batch, num_tokens, embed_dim)

        Q = self.W_q(x)       #Output Shape:(batch, num_tokens, d_out)
        #Now let's project the Input X to C_kv_latent vector.
        #(batch, num_tokens, embed_dim) --> (batch, num_tokens, latent_dim)
        C_kv_latent_new = self.W_kv(x)  

        #Now let's implement KV Cache on this Latent Vector.
        if kv_cache:
            # pass
            if self.c_kv_cache is None :
                self.c_kv_cache = C_kv_latent_new
            else:
                self.c_kv_cache = torch.cat(tensors=[self.c_kv_cache, C_kv_latent_new], dim=1)
            
            C_kv_latent_total = self.c_kv_cache

        else:
            # pass
            C_kv_latent_total = C_kv_latent_new
            self.c_kv_cache = None

        #Now we can upsample the dim_size and retreive the Keys & Values Vectors from cached C_kv_latent_total
        #(batch, num_tokens, latent_dim) -- > (batch, num_tokens, d_out)
        K = self.W_ku(C_kv_latent_total)
        V = self.W_vu(C_kv_latent_total)

        #Let's break the d_out dim into multiple heads & also transposing the num_tokens & num_heads dimension
        #(batch, num_tokens, d_out) -- > (batch, num_tokens, num_heads, head_dim) --> (batch, num_heads, num_tokens, head_dim)
        Q = Q.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1,2)
        K = K.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1,2)
        V = V.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1,2)

        attn_scores = Q @ K.transpose(2,3)

        ######### Applying Causal Attention Mask #######
        num_tokens_Q = Q.shape[-2]
        num_tokens_K = K.shape[-2]
        device = Q.device
        if kv_cache:
            #Output SHape:(num_tokens_Q, )
            Q_positions = torch.arange(
                start= self.ptr_current_pos,
                end = self.ptr_current_pos + num_tokens_Q,
                dtype = None,
                device = device
            )
            self.ptr_current_pos += num_tokens_Q
        else:
            #Output SHape:(num_tokens_Q, )
            Q_positions = torch.arange(self.ptr_current_pos + num_tokens_Q, dtype=torch.long, device=device)
            self.ptr_current_pos = 0

        #Output SHape:(num_tokens_K, )
        K_positions = torch.arange(num_tokens_K, dtype=torch.long, device=device)
        # (num_tokens_Q, 1) < (1, num_tokens_K) --> (num_tokens_Q, num_tokens_K) Matrix of comparison.
        causal_mask = Q_positions.unsqueeze(-1) < K_positions.unsqueeze(0)

        attn_scores.masked_fill_(causal_mask, -torch.inf)
        attn_weights = torch.softmax(input=attn_scores/K.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        #num_tokens_V == num_tokens_K
        #(batch, num_heads, num_tokens_Q, num_tokens_K) @ (batch, num_heads, num_tokens_V, head_dim) --> (batch, num_heads, num_tokens_Q, head_dim)
        context_vectors = attn_weights @ V
        context_vectors = context_vectors.transpose(1,2)   #(batch, num_tokens_Q, num_heads, head_dim)
        context_vectors = context_vectors.contiguous().view(batch_size, num_tokens, self.d_out)

        context_vectors = self.out_proj(context_vectors)

        return  context_vectors

    def reset_cache(self):

        self.c_kv_cache = None
        self.ptr_current_pos = 0




#Naive/Simple KV Cache Implementation on Transformer Block.
class TransformerBlockMLACachedSimple(nn.Module):

    def __init__(self, cfg):
        super().__int__()

        self.attn = MultiHeadLatentAttentionCachedSimple(
            d_in = cfg['emb_dim'],
            d_out = cfg['emb_dim'],
            context_length = cfg['context_length'],
            num_heads = cfg['n_heads'],
            latent_dim=cfg["latent_dim"],
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





#Gpt Model Class with MLA & Naive KV Cache Implementation.
class GPTModelMLACachedSimple(nn.Module):
    '''KV Cache Implementation GPT Model Class with Multi Head Latent Attention.
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
            [TransformerBlockMLACachedSimple(cfg) for _ in range(cfg['n_layers'])]
        )

        #We also need a Position Counter here as well for tracking for many tokens have been cached.
        self.current_pos = 0

        self.final_norm = LayerNorm(embed_dim=cfg['emb_dim'])
        self.dropout = nn.Dropout(p=cfg['drop_rate'])
        self.out_head = nn.Linear(in_features=cfg['emb_dim'], out_features=cfg['vocab_size'], bias=False)

    def forward(self, in_idx, kv_cache=False):

        batch_size, seq_len = in_idx.shape

        token_emb = self.token_emb(in_idx)    #(batch_size,seq_length, embed_dim)

        #Newly Modified Part.
        if kv_cache:
            pos_ids = torch.arange(start=self.current_pos, end=self.current_pos+seq_len, device=in_idx.device, dtype=torch.long)

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
        self.current_pos = 0




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





