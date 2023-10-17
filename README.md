# Two head transformer weights by hand
making GPT2 transformer weights by hand

I found this cool post about [making the weights of a transformer by hand](https://vgel.me/posts/handmade-transformer
) and thought "what about adding another head with a different rule?" 

[This code is for a 2 head transformer, by hand](/2heads-handmade-transformer.py)

# Approach

The starting place is to come up with an idea for the new head. My original idea was to add a new token 'c' and make a rule like "if previous sequence is aabaab -> 'c'. 

However, this is too complex to reasonably do by hand. It's trivial to get the attention matrix to 'pay attention' to 6 previous tokens, but then assigning the weights so that we know the exact sequence of a's and b's doesn't have a clean mechanism. Right now we're only looking at two tokens and taking the combination of those to make a one-hot encoding.  

Think about 3 tokens for a moment. With the mechanic in place we are simply providing a score to each a or b. So we can know that 'aab' is 2 a's and 1'b but we can't do anything about the order. So, we can't easily make a rule that differentiates 'aba' from 'aab'. This is obvious in retrospect but it took me awhile to realize that there isn't an easy way to differentiate 'aba' from 'aab'. I experiemented with this a lot and went down many dead ends. I suppose it's possible if we continue to expand the embedding size to account for all the permutations, but doesn't seem like a useful learning exercise so I didn't pursue it. 

If we can't get 3 tokens with positional tracking, forget 6.

So I had to change the rule to something that doesn't care about order, only looks at 2 tokens and with only a combination of a and b in any order can lead to a prediction. 

New rule: 
if current token is 'a' and 3rd previous token is 'a' then -> 'b'. Otherwise 'a'.

You can see an implementation of this rule alone in 'new-head.py'

The combined heads are in '2heads-handmade-transformer.py'

The combined heads given 'aab' will complete to 'aababaababaababaab', demonstrating the new 2nd head works, cool!


# Other changes

* increased the WPE to have longer context of 6

* unpacked attn matrix into q, k, v matricies and make them their own parts of the model. The packing in the model is for convenience, but it becomes unweildly as we add a 2nd head. This approach is much more clear. 

* removed attn['b'] which was just 0's that didn't do anything, simplified the code a bit

* increase head dimension from 8 to 10. one position is for WTE 'c' and one position is 
for the increased context size (WPE)

* simplified the MODEL structure a bit

* unpacked the gpt2() function to do everything in line for maximum readability. 

# Lessons

* In GPT2 the width of the q, k and v matrix is == N_EMBED. This is a coincidence, not a requirement. In this exercise it's good to keep them the same width otherwise we will lose either the position or the token encoding information, however in a real transformer, where we're doing compression all over the place, I think we can experiement with sizes that are not equivalent. I wouldn't have learned to question the model deeply enough to uncover this lesson if not for this exercise. 

    As an outcome, I think it's more clear to present the model weights and add the new variable ATTN_SIZE = N_HEADS * N_HEAD_DIM to clarify what fundamental constants drive the model dimensions: 

```
from pprint import pprint

N_HEAD_DIM = 9 
N_HEADS = 2
N_EMBED = 9
ATTN_SIZE = N_HEADS * N_HEAD_DIM

gpt2shape = {
    "wpe": [N_CTX, N_EMBED],  
    "wte": [N_VOCAB, N_EMBED],
    "ln_f": {"b": [N_EMBED], "g": [N_EMBED]},
    "blocks": [
        {
            "attn": {
                "c_attn": {"b": [ATTN_SIZE*3], "w": [N_EMBED, ATTN_SIZE*3]},
                "c_proj": {"b": [N_EMBED], "w": [ATTN_SIZE, N_EMBED]},
            },
            "ln_1": {"b": [N_EMBED], "g": [N_EMBED]},
            "ln_2": {"b": [N_EMBED], "g": [N_EMBED]},
            "mlp": {
                "c_fc": {"b": [N_EMBED*4], "w": [N_EMBED, N_EMBED*4]},
                "c_proj": {"b": [N_EMBED], "w": [N_EMBED*4, N_EMBED]},
            },
        }
    ]
}
pprint(gpt2shape, width=20)
```


* the c_proj matrix is where each attention head gets added back together. We can vertically split this matrix and see each head's contribution to the residual stream / prediction. Cool. 