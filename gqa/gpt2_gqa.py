import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tiktoken
from torch.utils.data import Dataset, DataLoader


'''

This Script contains Group-Query Attention Implementation on GPT-2 Architecture along with Naive KV Cache.

'''




#Test GPT2 Configuration.
#GPT2 Configuration.
GPT_CONFIG_124M = {
    "vocab_size": 50257, # Vocabulary size
    "context_length": 256, # Context length  #Change to 256 from 1024 for Learning
    "emb_dim": 768, # Embedding dimension
    "n_heads": 12, # Number of attention heads
    "n_layers": 12, # Number of layers i.e specifies the number of Transformer blocks in the model
    'num_kv_groups': 3, 
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




#Simple/Naive KV Cache Based Grouped Query Attention Implementation.
class GroupedQueryAttentionCachedSimple(nn.Module):
    '''Class Implementation of Grouped Query Attention along with KV Cache.
    Note: This One has used Modified Causal Masking Techniques & Also Two Types of Grouped Attention Calculation strategy has been shown here.'''

    def __init__(self, d_in, d_out, context_length, num_heads, num_kv_groups, dropout=0.1, qkv_bias=False, dtype=None):
        super().__init__()

        device = 'cpu'

        assert d_out % num_heads == 0, 'd_out must be divisible by num_heads'
        assert num_heads % num_kv_groups == 0, 'num_heads must be divisible by num_kv_groups'

        self.head_dim, self.num_heads, self.context_length = d_out // num_heads, num_heads, context_length
        self.num_kv_groups = num_kv_groups
        self.d_in, self.d_out = d_in, d_out

        self.W_q = nn.Linear(in_features=d_in, out_features=d_out, bias=qkv_bias, device=device, dtype=dtype)  

        #For W_k &W_v matrices we output_dim is not d_out rather it is (self.num_kv_groups * self.head_dim) 
        #The reason for that is we need to calculate the attn_weights by matrix multiplication based on Query's head_dim & Key's head_dim.
        #So we keep the same head_dim but variable no. of key, value groups.
        #Because if we did kv_dim = d_out // num_kv_groups we would not be able to matrix multiply Query & Value vectors as their embed_dim would be different.
        self.W_k = nn.Linear(d_in, num_kv_groups*self.head_dim, bias=qkv_bias, device=device, dtype=dtype)
        self.W_v = nn.Linear(d_in, num_kv_groups*self.head_dim, bias=qkv_bias, device=device, dtype=dtype)
        #The following line gives how many heads per Key, Value Pair or how many heads per KV pair.
        self.group_size = self.num_heads // num_kv_groups

        self.dropout = nn.Dropout(p=dropout)  #This dropout is used in attention weight dropping during training.
        self.out_proj = nn.Linear(in_features=d_out, out_features=d_out, bias=None, device=device, dtype=dtype)

        #Note: Here we haven't used a register buffer for causal_mask as we will use different more robust implementation of Causal Masking in forward.
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
        #This one is used in KV Caching to track number of Cached Tokens.
        self.ptr_current_pos = 0  


    def forward(self, x, kv_cache=False):

        batch_size, num_tokens, d_in = x.shape

        Q = self.W_q(x)  #Output SHape:(batch_size, num_tokens, d_out) or if kv_cache enabled then (batch_size, Q_num_tokens, d_out)
        K_new = self.W_k(x) #Output Shape:(batch_size, num_tokens, num_kv_groups*self.head_dim) or if kv_cache enabled then (batch_size, K_num_tokens, num_kv_groups*self.head_dim)
        V_new = self.W_v(x) #Output Shape:(batch_size, num_tokens, num_kv_groups*self.head_dim)

        #Let's Reshape.
        #Breaking d_out into (num_heads, head_dim) & transposing num_token, num_heads in same line.
        #Output Shape:(batch_size, num_heads, num_tokens, head_dim)
        Q = Q.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1,2) 
        #Breaking Key & Value Tensors into (self.num_kv_groups, self.head_dim) & transposing num_token, num_kv_groups in same line.
        K_new = K_new.view(batch_size, num_tokens, self.num_kv_groups, self.head_dim).transpose(1,2) 
        V_new = V_new.view(batch_size, num_tokens, self.num_kv_groups, self.head_dim).transpose(1,2) 

        if kv_cache:
            # pass
            if self.k_cache is None or self.v_cache is None:
                # pass
                self.k_cache, self.v_cache = K_new, V_new
            else:
                K = torch.cat(tensors=[self.k_cache, K_new], dim=2)  #Concating along the num_tokens dimension
                V = torch.cat(tensors=[self.v_cache, V_new], dim=2)
            #Updating the KV Cache with new tokens.
            #Once we have got the old Keys & Values concatenated withnew Ones now time for Attention Calculation.
            K, V = self.k_cache, self.v_cache
        else:
            #As we are not using KV Cache so we can directly assign.
            K, V = K_new, V_new

        ####### Newly Modified Part for Grouped Query ATtention #####
        #Checkout torch.tensor.repeat_interleave(): https://docs.pytorch.org/docs/2.14/generated/torch.repeat_interleave.html
        # Expanding keys and values to match the number of heads
        #Since we have Query dim=1 as self.num_heads So to copy the Key & Value tensors as same size so we have to do this.
        #Input Shape:(batch, num_kv_groups, K_num_tokens, head_dim)  --> Output Shape:(batch, num_heads, K_num_tokens, head_dim)
        K = K.repeat_interleave(repeats=self.group_size, dim=1)  
        V = V.repeat_interleave(repeats=self.group_size, dim=1)
        # For example, before repeat_interleave along dim=1 (query groups):
        #   [K1, K2]
        # After repeat_interleave (each query group is repeated group_size times):
        #   [K1, K1, K2, K2]
        # If we used regular repeat instead of repeat_interleave, we'd get:
        #   [K1, K2, K1, K2]

        #Note: Instead of using repeat_interleave we can use broadcasting in pytorch.
        #Broadcasting Method for Attention (Alternative): 
        # Q.view(batch_size, self.num_kv_groups, self.group_size, Q_num_tokens, self.head_dim) @ K.view(batch_size, self.num_kv_groups, 1, K_num_tokens, self.head_dim).transpose(2,3)

        #Scaled dot product Attention.
        attn_scores = Q @ K.transpose(2,3)   #Output Shape:(batch, num_heads, Q_num_tokens, K_num_tokens)
        #Q_num_tokens is the number of query tokens & K_num_tokens is the cached+new set of tokens.


        ######## Newly Modified Causal Masking After Attention Score Calculation. ##########
        #Alternative & Better Robust way to Calculate Causal Masking when Query_length & Total Ccahed Key, Value length is different.
        num_tokens_Q = Q.shape[-2]
        num_tokens_K = K.shape[-2]
        device = Q.device
        if kv_cache:
            # pass
            #Output SHape:(num_tokens_Q,)
            Q_positions = torch.arange(
                start = self.ptr_current_pos,
                end= self.ptr_current_pos + num_tokens_Q,
                dtype=None,
                device=device
            )
            #Update the tracking variable till latest cached num_tokens.
            self.ptr_current_pos += num_tokens_Q
        else:
            #Output SHape:(num_tokens_Q,)
            Q_positions = torch.arange(
                end= self.ptr_current_pos + num_tokens_Q,
                dtype=None,
                device=device
            )
            self.ptr_current_pos = 0  #As in this case no KV Cache stored so points to zero.
            #Output SHape:(num_tokens_K,)
        K_positions = torch.arange(end=num_tokens_K, dtype=None, device=device)

        #### ALternative Masking Dynamically unlike creating static Upper Triangular Matrices.
        # [[num_tokens_Q],] i.e (num_tokens_Q, 1) < [1, [num_tokens_K]] i.e (1, num_tokens_K) --> Gives Matrix of (num_tokens_Q, num_tokens_K)
        #This happens as a result of broadcasting in Pytorch.
        mask = Q_positions.unsqueeze(-1) < K_positions.unsqueeze(0)  

        #Example:
        '''
        Suppose : Q_positions = [5, 6, 7] (Shape:(3,) or [3]) & K_positions = [0,1,2,3,4,5,6,7]    shape [8]
        So Q_positions.unsqueeze(-1) = [[5],[6],[7]] (Shape:(3,1)) & K_positions.unsqueeze(0) = [[0,1,2,3,4,5,6,7]] (Shape:((1,8)))
        And comparing (3,1) & (1,8) allowed broadcasting so became a matrix of shape:(3,8)
                   
            Q            K →
            ↓   0  1  2  3  4  5  6  7
            5   5   5  5  5  5  5  5  5  
            6   6  6  6  6  6  6  6  6
            7   7  7  7  7  7  7  7  7
                ↓
                compare
        
        Hence since Q_pos & K_pos were positional indices so for position 5 we cant calculate Attention beyond 5 so it should be False mask from 5 to 7 and so on.
        So the comparison becomes:

            5 < [0,1,2,3,4,5,6,7]
            6 < [0,1,2,3,4,5,6,7]
            7 < [0,1,2,3,4,5,6,7]

        giving:

            [[False, False, False, False, False, False, True,  True ],
            [False, False, False, False, False, False, False, True ],
            [False, False, False, False, False, False, False, False]]
        
        '''

        attn_scores.masked_fill_(mask, -torch.inf)

        assert K.shape[-1] == self.head_dim, 'K embedding dim should be same as Each Head Embedding Dimension'
        attn_weights = torch.softmax(input=attn_scores / K.shape[-1]**0.5, dim=-1) #Output Shape :(batch, num_heads, Q_num_tokens, K_num_tokens)

        #Since Keys & Values are of same dim i.e K_num_tokens = V_num_tokens
        #(batch, num_heads, Q_num_tokens, K_num_tokens) @ (batch, num_heads, V_num_tokens, head_dim) --> (batch, num_heads, Q_num_tokens, head_dim)
        context_vectors = attn_weights @ V 
        context_vectors = context_vectors.transpose(1,2)    #Output Shape:(batch, Q_num_tokens, num_heads, head_dim)

         # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vectors = context_vectors.contiguous().view(batch_size, num_tokens, self.d_out)
        context_vectors = self.out_proj(context_vectors)

        return  context_vectors

    def reset_cache(self):
        self.cache_k, self.cache_v = None, None
        self.ptr_current_pos = 0





#Naive/Simple KV Cache Implementation on Transformer Block.
class TransformerBlockGQACachedSimple(nn.Module):

    def __init__(self, cfg):
        super().__int__()

        self.attn = GroupedQueryAttentionCachedSimple(
            d_in = cfg['emb_dim'],
            d_out = cfg['emb_dim'],
            context_length = cfg['context_length'],
            num_heads = cfg['n_heads'],
            num_kv_groups = cfg['num_kv_groups'],
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



#Gpt Model Class with GQA & Naive KV Cache Implementation.
class GPTModelGQACachedSimple(nn.Module):
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
            [TransformerBlockGQACachedSimple(cfg) for _ in range(cfg['n_layers'])]
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



