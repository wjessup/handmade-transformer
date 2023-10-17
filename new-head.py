'''
I used this file to build and debug the new head

Rule:
if current token is 'a' and 3rd previous token is 'a' then -> 'b'. Otherwise 'a'.

'''

import numpy as np
N_CTX = 5
N_VOCAB = 2
N_HEAD_DIM = 9 
N_HEADS = 2
N_EMBED = 9
ATTN_SIZE = N_HEADS * N_HEAD_DIM
Lg = 1024  
np.set_printoptions(suppress=True, precision=2, linewidth=100)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def attention(q, k, v):
    print("q = \n", q)
    print("k = \n", k)
    print("k.T = \n", k.T)
    print(f"q @ k.T  =\n", (q @ k.T) )
    print(f"attn scores =\n", softmax(q @ k.T) )
    print(f"v = \n", v)

    return softmax(q @ k.T) @ v

def gpt(inputs, wte, wpe, c_attn, c_proj):

    x = wte[inputs] + wpe[range(len(inputs))]  
    print("x = \n", x)
  
    q = x @ c_attn['q']
    k = x @ c_attn['k']
    v = x @ c_attn['v']

    attn_out = attention(q, k, v)
    print(f"attn_out 1 =\n", attn_out)
    attn = attn_out @ c_proj['w'] + c_proj['b'] 

    
    print("1: attn_out @ c_proj + bias = \n", attn)

 

    x = x + attn 

    print("x + attn = \n", x)
    print("wte.T)= \n", wte.T)
    print("x @ wte.T = \n", x @ wte.T)
    return x @ wte.T 



MODEL = {
    "wte": np.array(
        # one-hot token embeddings
        [
            [0, 0, 0, 0, 0, 0, 1, 0],  # token `a` (id 0)
            [0, 0, 0, 0, 0, 0, 0, 1]   # token `b` (id 1)
        ]
    ),
    "wpe": np.array(
        # one-hot position embeddings
        [
            [1, 0, 0, 0, 0, 0, 0, 0],  # position 0
            [0, 1, 0, 0, 0, 0, 0, 0],  # position 1
            [0, 0, 1, 0, 0, 0, 0, 0],  # position 2
            [0, 0, 0, 1, 0, 0, 0, 0],  # position 3
            [0, 0, 0, 0, 1, 0, 0, 0],  # position 4
            [0, 0, 0, 0, 0, 1, 0, 0]   # position 5
        ]
    ), 
    "c_attn": { 
        "q" : np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0], 
                [0, 0, 0, 0, 0, 0, 0, 0], 
                [0, 0, 0, 0, 0, 0, 0, 0],
                [8, 0, 0, 8, 0, 0, 0, 0],
                [0, 8, 0, 0, 8, 0, 0, 0],
                [0, 0, 8, 0, 0, 8, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ]
        ),
        # aabaa
        # rule:
        # if current is 'a'
        # and 3 ahead is 'a'
        # print 'b'. 
        "k" : np.array(
            [
                
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0], 
                [0, 0, 1, 0, 0, 0, 0, 0], 
                [0, 0, 0, 1, 0, 0, 0, 0], 
                [0, 0, 0, 0, 1, 0, 0, 0], 
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1]
            ]  
        ),
        "v" : np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0], 
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, -1]
            ]
        )
    },
    "c_proj": {  # weights to project attn result back to embedding space
        "b": [0, 0, 0, 0, 0, 0, 8, 0],
        "w": np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, -8, 8],
            ]
        ),
        
    },
}



CHARS = ["a", "b"]
def tokenize(s): return [CHARS.index(c) for c in s]
def untok(tok): return CHARS[tok]

def predict(s):
    tokens = tokenize(s)[-6:]
    logits = gpt(np.array(tokens), **MODEL)


    pred = np.argmax(logits[-1])
    
    print("pred = ", untok(pred))


    return np.argmax(logits[-1])


print("aababb")
predict("aababb")
