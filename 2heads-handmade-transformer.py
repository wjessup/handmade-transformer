#run with 'python3 2heads-handmade-transformer.py'

import numpy as np
N_CTX = 5
N_VOCAB = 2
N_HEAD_DIM = 9 
N_HEADS = 2
N_EMBED = 9
ATTN_SIZE = N_HEADS * N_HEAD_DIM
np.set_printoptions(suppress=True, precision=2, linewidth=100)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def attention(i, q, k, v):
    return softmax((q @ k.T) ) @ v

def gpt(inputs, wte, wpe, c_attn, c_proj):
    x = wte[inputs] + wpe[range(len(inputs))]  
    q = x @ c_attn['q']
    k = x @ c_attn['k']
    v = x @ c_attn['v']
    
    q_heads = np.split(q, N_HEADS, axis=-1)
    k_heads = np.split(k, N_HEADS, axis=-1)
    v_heads = np.split(v, N_HEADS, axis=-1)
    c_proj_heads = np.split(c_proj['w'], N_HEADS, axis=0)
    
    attn_out = [attention(i, q_heads[i], k_heads[i], v_heads[i]) for i in range(N_HEADS)]
    
    attn0 = attn_out[0] @ c_proj_heads[0] + c_proj['b'] 
    attn1 = attn_out[1] @ c_proj_heads[1] + c_proj['b'] 
    attn = attn0 + attn1
    
    x = x + attn 
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
                [8, 0, 0, 0, 0, 0, 0, 0,    0, 0, 0, 0, 0, 0, 0, 0], 
                [8, 8, 0, 0, 0, 0, 0, 0,    0, 0, 0, 0, 0, 0, 0, 0], 
                [0, 8, 8, 0, 0, 0, 0, 0,    0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 8, 8, 0, 0, 0, 0,    8, 0, 0, 8, 0, 0, 0, 0],
                [0, 0, 0, 8, 8, 0, 0, 0,    0, 8, 0, 0, 8, 0, 0, 0],
                [0, 0, 0, 0, 8, 8, 0, 0,    0, 0, 8, 0, 0, 8, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0,    0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0,    0, 0, 0, 0, 0, 0, 0, 0],
            ]
        ),
        "k" : np.array(
            [
                
                [1, 0, 0, 0, 0, 0, 0, 0,    1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0,    0, 1, 0, 0, 0, 0, 0, 0], 
                [0, 0, 1, 0, 0, 0, 0, 0,    0, 0, 1, 0, 0, 0, 0, 0], 
                [0, 0, 0, 1, 0, 0, 0, 0,    0, 0, 0, 1, 0, 0, 0, 0], 
                [0, 0, 0, 0, 1, 0, 0, 0,    0, 0, 0, 0, 1, 0, 0, 0], 
                [0, 0, 0, 0, 0, 1, 0, 0,    0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0,    0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0,    0, 0, 0, 0, 0, 0, 0, 1]
            ]  
        ),
        "v" : np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0,    0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0,    0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0,    0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0,    0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0,    0, 0, 0, 0, 0, 0, 0, 0], 
                [0, 0, 0, 0, 0, 0, 0, 0,    0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1,    0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, -1,    0, 0, 0, 0, 0, 0, 0, -1]
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

                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, -9, 9],
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

def complete(s, max_new_tokens=15):
    tokens = tokenize(s)
    while len(tokens) < len(s) + max_new_tokens:
        logits = gpt(np.array(tokens[-6:]), **MODEL)
        probs = softmax(logits)
        pred = np.argmax(probs[-1])
        tokens.append(pred)
    return s + "".join(untok(t) for t in tokens[len(s):])

#print("aabaab")
#predict("aabab")

print(complete("aab"))
