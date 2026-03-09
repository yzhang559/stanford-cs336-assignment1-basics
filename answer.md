# 2 Byte-Pair Encoding (BPE) Tokenizer

chr() returns one char string that corresponds to the given unicode code point, inverse function of ord()
__repr__() returns a string that can be used to recreate the object, stable while the printed representation is invisible
and can cause issues in terminal output

Problem (unicode2): Unicode Encodings (3 points)
Why UTF8 is the best encoding for most use cases?
UTF8 gives best balance of compactness, vocabulary size, and compatibility (first 128 characters are the same as ASCII)
For "Hello 世界" the UTF8 encoding size is 12 bytes, while UTF16 is 18 bytes and UTF32 is 36 bytes

Two bytes that does not decode to any Unicode 
Some char need to be represented with multiple bytes, like Chinese characters, decoding them separately will cause
UnicodeDecodeError: 'utf-8' codec can't decode byte 0x95 in position 0: invalid start byte

Problem (tokenizer_experiments): Experiments with tokenizers (4 points)
The processing time of tinystories is 
```
time: 108.72s, peak RSS: 157.52 MB
Longest token:  accomplishment, length: 15, raw bytes: b' accomplishment'
```

For 2.1GB, the throughput is 19.3 MB/s.

---

# 3 Transformer Language Model Architecture
(a) Consider GPT-2 XL, which has the following configuration:
vocab_size : 50,257
context_length : 1,024
num_layers : 48
d_model : 1,600
27
num_heads : 25
d_ff : 6,400
Suppose we constructed our model using this configuration. How many trainable parameters
would our model have? Assuming each parameter is represented using single-precision floating
point, how much memory is required to just load

1. token embedding
E(vocab_size, d_model) = 80,411,200
2. transformer blocks
RMSNorm: (d_model, ) = 1600
MultiHeadSelfAttention: W_Q, W_K, W_V, W_O (d_model, d_model) = 4 * d_model * d_model = 10,240,000
RMSNorm (PRE-FFN): (d_model, _) = 1600
FFN: W1 (d_model, d_ff), W3 (d_model, d_ff), W2 (d_ff, d_model) = 3 * d_model * d_ff = 3,072,000
3. Final RMSNorm: (d_model, ) = 1600
4. LM Head (output layer): d_model * vocab_size = 80,411,200

80,411,200 * 2 + 1600 + 48 * (1600+10,240,000+1600+3,072,000) = 160,822,400 + 1600 + 48 * 40,963,200 = 2,127,057,600
Suppose each parameter is float 32, 4 bytes * 2,127,057,600 = 8,508,230,400 bytes = 8.5 GB

(b) Identify the matrix multiplies required to complete a forward pass of our GPT-2 XL-shaped
model. How many FLOPs do these matrix multiplies require in total? Assume that our input
sequence has context_length tokens.

Summary of Matrix Multiplies
Per Layer (×48):

W_Q projection: 2 × T × d_model²
W_K projection: 2 × T × d_model²
W_V projection: 2 × T × d_model²
QK^T (attention scores): 2 × T² × d_model
Attention @ V: 2 × T² × d_model
W_O projection: 2 × T × d_model²
W1 (FFN): 2 × T × d_model × d_ff
W3 (FFN): 2 × T × d_model × d_ff
W2 (FFN): 2 × T × d_ff × d_model
Final Layer: 10. LM Head: 2 × T × d_model × vocab_size

Total: ~4.51 TFLOPs per forward pass with sequence length 1,024

(c) Based on your analysis above, which parts of the model require the most FLOPs?
From the GPT-2 XL analysis:

Attention per layer: 27,682,406,400 FLOPs (30.5%)
FFN per layer: 62,914,560,000 FLOPs (69.5%)
All 48 layers: 4,348,654,387,200 FLOPs (96.4%)
LM Head: 164,679,270,400 FLOPs (3.6%)

(d) Repeat your analysis with GPT-2 small (12 layers, 768 d_model, 12 heads), GPT-2 medium (24
layers, 1024 d_model, 16 heads), and GPT-2 large (36 layers, 1280 d_model, 20 heads). As the
model size increases, which parts of the Transformer LM take up proportionally more or less of
the total FLOPs?
Deliverable: For each model, provide a breakdown of model components and its associated
FLOPs (as a proportion of the total FLOPs required for a forward pass). In addition, provide a
one-to-two sentence description of how varying the model size changes the proportional FLOPs
of each component.

| Model | Attention | FFN | LM Head | Total FLOPs |
|-------|-----------|-----|---------|-------------|
| Small | 32.1% | 51.9% | 23.5% | 335 GFLOPs |
| Medium | 26.3% | 63.1% | 10.7% | 981 GFLOPs |
| Large | 26.0% | 67.8% | 6.2% | 2.14 TFLOPs |
| XL | 29.5% | 67.0% | 3.6% | 4.51 TFLOPs |

As model size increases, the FFN consistently dominates (growing from 52% to 67%), 
while the LM head's proportion decreases dramatically (from 24% to 4%) because it scales with 
d_model while layers scale with both d_model² and number of layers. Attention remains relatively 
stable at 26-32% across all model sizes.

(e) Take GPT-2 XL and increase the context length to 16,384. How does the total FLOPs for one
forward pass change? How do the relative contribution of FLOPs of the model components
change?
Change: 149.5 / 4.51 ≈ 33.1× increase (context length increased 16×)

Attention: 46.8% (was 29.5%)
FFN: 51.4% (was 67.0%)
LM Head: 1.8% (was 3.6%)
Total FLOPs increase by ~33× (from 4.51 to 149.5 TFLOPs) when context length increases 16×, 
because attention's quadratic operations (QK^T and Attention@V) dominate at longer sequences. 
The attention component grows from 29.5% to 46.8% of total FLOPs, while FFN drops from 67% to 51% and 
LM head drops from 3.6% to 1.8%, showing that attention becomes the bottleneck at long context lengths.

