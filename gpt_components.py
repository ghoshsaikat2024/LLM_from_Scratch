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





#Attention Class
#Causal MuliHeadAttention Class from Previous Chapter.
class MultiHeadAttention(nn.Module):

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
            tensor=torch.triu(input=torch.ones(size=(context_length, context_length)), diagonal=1)
        )

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(in_features=d_out, out_features=d_out)

    def forward(self, x):

        #x or input is of shape (batch_size, seq_length, d_in)
        batch_size, num_tokens, d_in = x.shape
        Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)  #Using a Giant W_q/W_k/W_v matrix to create (batch_size, seq_length, d_in) ->(batch_size, seq_length, d_out)
        #Now we need to  the Giant matrices so that (batch_size, seq_length, d_out) becomes (batch_size, seq_length, num_heads, head_dim).
        Q = Q.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        K = K.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        V = V.view(batch_size, num_tokens, self.num_heads, self.head_dim)

        #Now we need to change calculate Attention for each block separately so first we have to interchange the num_tokens & num_heads place.
        Q, K, V = Q.transpose(1,2), K.transpose(1,2), V.transpose(1,2)  #Shape Output: (batch_size, num_heads,  num_tokens, head_dim)

        attn_scores = Q @ K.transpose(2,3)  #Output Shape: (batch_size, num_heads,  seq_length, seq_length)

        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )

        attn_weights = torch.softmax(attn_scores/(K.shape[-1]**0.5), dim=-1)  #(batch_size, num_heads,  num_tokens , num_tokens)
        attn_weights = self.dropout(attn_weights)

        context_vectors = attn_weights @ V  #Output Shape: (batch_size, num_heads,  num_tokens, head_dim)
        #Reversing the position of num_heads & num_tokens
        context_vectors =  context_vectors.transpose(1,2)  #Output Shape: (batch_size, num_tokens, num_heads, head_dim)

        #Now we need to make the current context_vector shape memory contiguus and then couple the num_heads & head_dim into one.

        context_vectors = context_vectors.contiguous().view(batch_size, num_tokens, self.d_out)  #Merged the num_heads & head_dim into d_out.

        context_vectors = self.out_proj(context_vectors)
        return context_vectors




#Transformer Block
class TransformerBlock(nn.Module):
    """This class implments only the Transformer Block Part of th GPT Model.
        Follow the Illustration of the Transformer Block given above to implement it.
        Token , Positional , Linear Layer Projection are in different Blocks."""

    def __init__(self, cfg):
        super().__init__()
        self.CausalMultiHeadAttentionLayer = MultiHeadAttention(
            d_in = cfg['emb_dim'],
            d_out = cfg['emb_dim'],
            context_length = cfg['context_length'],
            num_heads = cfg['n_heads'],
            dropout = 0.1,
            qkv_bias = cfg['qkv_bias']
        )
        self.ffn = FeedForward(cfg)   #Using the FFN Layer we created in previous blocks
        self.norm1 = LayerNorm(embed_dim=cfg['emb_dim'])   #Using the LayerNorm Layer we created in previous blocks
        self.norm2 = LayerNorm(embed_dim=cfg['emb_dim'])
        self.dropout = nn.Dropout(cfg['drop_rate'])

    def forward(self, x):

        #Calling the x as shortcut connection.
        #Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.CausalMultiHeadAttentionLayer(x)
        x = self.dropout(x)
        x = x + shortcut

        #Shortcut connection for FFN block
        shortcut = x 
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + shortcut

        return x





#Compact Implementation of the GPT(Decoder) Model Class
class GPTModel(nn.Module):

    """This is the Compact Class GPT Model (Note: GPT is a Decoder Only Transformer Model)
        Follow and use the above Image of the GPT Model to code the whole GPT Model."""

    def __init__(self, cfg):
        super().__init__()
        #token_embedding_generation.
        self.token_emb = nn.Embedding(num_embeddings=cfg['vocab_size'], embedding_dim=cfg['emb_dim'])
        #Since Context Length is Max Input length so our Architecture can handle at most this distant Tokens 
        #There are only <context_length> possible positions & So we need one vector for each possible position:
        #Why NOT vocab_size × emb_dim for positional embeddings?
        # Because position has nothing to do with which token is there. 
        # The position embedding for position 5 should be the same irrespective of which token it is.
        self.pos_emb = nn.Embedding(num_embeddings=cfg['context_length'], embedding_dim=cfg['emb_dim'])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg['n_layers'])]
        )
        self.final_norm = LayerNorm(embed_dim=cfg['emb_dim'])
        self.dropout = nn.Dropout(p=cfg['drop_rate'])
        #Projection Layer Across the Vocab Size Dim.
        #PyTorch defines nn.Linear() Layer as: y = x @ W^T +b, So when we store W as nn.Linear(in_features=cfg['emb_dim'], out_features=cfg['vocab_size'])
        #W is stored as (50257, 768) even though we said (768, 50257) and internally when multiplied with x(batch_size, num_tokens, 768) they do (50257, 768).T i.e (batch_size, num_tokens, 768) @ (768, 50257).
        #So W_out_head can be thought of having same size of W_token_emb size.
        #Infact we can use the same Matrix for these things, which is called Weight Tying i.e to reuse the Token_Embedding Matrix in the Output Head as well.
        self.out_head = nn.Linear(in_features=cfg['emb_dim'], out_features=cfg['vocab_size'], bias=False)
        

    def forward(self, in_idx):  
        #in_dx is the token indices of a input like for a Input Every effort moves you -> Token Ids : [23,45,65,32,12]
        #Here after Tokenizing and encoding we pass these indices which the model takes and gives us the logits across the vocab_dim.
        #Then we take the max or use Beam Search or top K like techniques to get the token_ids and then from that get the decoded token from vocab_dict.
        
        batch_size, seq_length = in_idx.shape   #in_idx Shape: (batch_size, num_tokens)
        #Now let's get the token_embedings of the given token_indices.So we may get the embeddings of the token_ids of [23,45,65,32,12]
        #Making the shape from (batch_size, seq_length) to (batch_size, seq_length, emb_dim)
        token_embeddings = self.token_emb(in_idx)
        #We are getting the Positional Embedding from the Positions of the tokens starting from 0 to Seq_length for a Sequence.
        pos_embeddings = self.pos_emb(torch.arange(seq_length, device=in_idx.device))  #The device setting will allow us to train the model on a CPU or GPU, depending on which device the input data sits on. 

        x = token_embeddings + pos_embeddings #(batch_size, num_tokens, embed_dim)
        x = self.dropout(x)
        x = self.trf_blocks(x)  #(batch_size, num_tokens, embed_dim)
        x = self.final_norm(x)  #((batch_size, num_tokens, embed_dim))
        logits = self.out_head(x)   #(batch_size, num_tokens, vocab_dim)
        
        return  logits




def generate_text_greedy(model, input_idx, max_new_tokens, context_length):

    for _ in range(max_new_tokens):
        input_idx = input_idx[:, -context_length:]   #Input Indices is of Shape:(batch, num_token_ids)
        #Disables gradient tracking since we are not training yet
        with torch.no_grad(): # To avoid storing the computational Graph
            logits = model(input_idx)                   #Output SHape : (batch, num_tokens, vocab_dim)

        logits = logits[:, -1, :]                   #Output Shape: (batch, 1, vocab_dim)
        probas = torch.softmax(input=logits, dim=-1)
        next_token_id = torch.argmax(input=probas, dim=-1, keepdim=True)    #Output SHape:(batch, 1)

        input_idx = torch.cat(tensors=[input_idx, next_token_id], dim=-1)  #Input: cat[(batch, num_tokens), (batch, 1), dim=-1] ; Output Shape: (batch, num_tokens + 1)


    return  input_idx


#TopK, Temperature, Multinomial Implementation in the advanced generate_text function.
def generate_text_advance(model, idx, max_new_tokens, context_length, temperature=0, top_k=None,  eos_id=None):
    '''Function to Generate Advanced Token Generation Strategies.'''

    for _ in range(max_new_tokens) :
        with torch.no_grad():
            idx = idx[:,-context_length:]            #input_idx Shape:(batch, num_tokens)  Convert to Shape:(batch, context_length)
            logits = model(idx)                        #logits Shape:(batch,num_tokens, vocab_dim)
            logits = logits[:, -1, :]                   #logits Shape:(batch, vocab_dim)  If you want (batch, 1,vocab_dim) use a slice logits[:, -1:, :] as with index it only return that element and slice an iterable.

        #Extract top_k.
        if top_k is not None:
            #Filters logits with top_k sampling
            top_k_logits, top_k_pos = torch.topk(input=logits, k=top_k)  #torch.topk() can handle batch of inputs and for each input it seprately does topk. #Output Shape:(batch, top_k)
            min_val = top_k_logits[:,-1]
            logits = torch.where(
                condition = logits < min_val,     #For each row in batch taking the minimum element 
                input = torch.tensor(float('-inf')).to(logits.device),
                other = logits
            )
        if temperature > 0:
            logits = logits / temperature
            probas = torch.softmax(logits, dim=-1)              
            next_idx = torch.multinomial(input=probas, num_samples=1)       #Shape:(batch, 1) So for a batch of sequences (batch, num_tokens) given this is the batch_output.
        
        else:
            #Carries out greedy nexttoken selection as before when temperature scaling is disabled
            probas = torch.softmax(logits, dim=-1)
            next_idx = torch.argmax(input=probas, dim=-1)

        #Stops generating early if end-of-sequence token is encountered
        if next_idx == eos_id: #If EOS Token Id matches our next_token_id then exit generating sequences.
            break

        idx = torch.cat(tensors=[idx, next_idx], dim=-1)   #Input Shapes: (batch, num_tokens) & (bathc, 1) -> Output SHape: (batch, num_tokens+1)

    return  idx

    








    