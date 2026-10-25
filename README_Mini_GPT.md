# Mini GPT --- Introduction Guide

A concise, visual guide to how this from-scratch GPT-style language
model works, written in plain English for quick revision.

> **One-line memory:** Text → Numbers → Transformer → Predict → Learn
> from error → Generate text

## 1. What did I build?

A small GPT-style language model built with **Python and PyTorch**. It
learns patterns from a text file and generates text by predicting the
next character.

This is a learning project designed to understand the core ideas behind
GPT---not a production-scale LLM.

## 2. The whole model

### Training

``` text
input.txt
   |
   v
Character tokenizer  ---> Characters become IDs
   |
   v
Training examples    ---> Input and next-character target
   |
   v
Token + position embeddings
   |
   v
Transformer blocks (6 configured)
   |
   v
Logits: scores for possible next characters
   |
   v
Compare prediction with correct answer
   |
   v
Loss -> Backpropagation -> AdamW updates weights
   |
   +------------------- repeat --------------------+
```

### Generation

``` text
Starting text -> Predict next character -> Append it
                      ^                       |
                      |_______________________|
                                repeat
```

**Remember:** During training, the model learns from known examples.
During generation, it uses what it learned to produce text.

## 3. Tokenization --- the translator

This project currently uses **character-level tokenization**. Each
unique character gets an ID.

``` text
"hello" -> [h, e, l, l, o] -> [IDs...]
```

-   `stoi`: character → ID
-   `itos`: ID → character

**Recall:** The model processes IDs, not raw text.

### How BPE would differ

``` text
Character tokens:  ["p", "l", "a", "y", "i", "n", "g"]
BPE tokens:        ["play", "ing"]  (illustrative)
```

BPE changes how text is split into tokens; the Transformer architecture
can remain the same.

## 4. Training example --- learn what comes next

``` text
Text:    h  e  l  l  o
Input:   h  e  l  l
Target:  e  l  l  o
```

The target is shifted one position to the right. The model learns to
predict the next character from the characters before it.

-   **Training data:** examples the model learns from.
-   **Validation data:** held-out examples used to check how well it
    learned.

## 5. Embeddings --- learned representations

``` text
Token ID ------> [0.21, -0.43, 0.72, ...]
Position ------> information about where the token appears
```

-   **Token embedding = what token it is.**
-   **Position embedding = where it appears.**

The vectors are learned during training.

## 6. Transformer block --- the core

``` text
Input
  |
  v
Causal self-attention  -> Weigh useful earlier context
  |
  v
Residual + LayerNorm   -> Help information flow and training stay stable
  |
  v
Feed-forward network  -> Further process each position's representation
  |
  v
Residual + LayerNorm
  |
  v
Output -> next block (repeat)
```

This project configures **6 attention heads** and **6 Transformer
blocks**. Each attention head can learn a different way to relate
tokens.

## 7. Self-attention --- what should I pay attention to?

Example: `The dog chased the ball because it ...`

Attention helps the model weigh which earlier tokens are useful for the
next prediction.

-   **Q --- Query:** What information am I looking for?
-   **K --- Key:** What information does this token offer?
-   **V --- Value:** What information gets passed along?

**Recall:** Q asks, K matches, V carries information.

## 8. Causal mask --- no cheating

When predicting left to right, a token must not see future tokens.

  -----------------------------------------------------------------------
  Token        Can see token  Can see token  Can see token  Can see token
                           1              2              3              4
  ----------- -------------- -------------- -------------- --------------
  1                        ✓              ✗              ✗              ✗

  2                        ✓              ✓              ✗              ✗

  3                        ✓              ✓              ✓              ✗

  4                        ✓              ✓              ✓              ✓
  -----------------------------------------------------------------------

**Recall:** Past and present are allowed; the future is blocked.

## 9. Prediction and learning

``` text
Transformer output
       |
       v
Logits = raw scores for possible next characters
       |
       v
Softmax = scores converted into probabilities
       |
       v
Compare with the correct next character
       |
       v
Cross-entropy loss = how wrong the prediction was
       |
       v
Backpropagation = calculate gradients
       |
       v
AdamW = update the weights
```

**Recall:** - **Loss:** measures prediction error. -
**Backpropagation:** calculates how parameters should change. -
**AdamW:** uses gradients to update the parameters.

## 10. Generation --- how text appears

``` text
Prompt: "The"
    |
    v
Predict next character
    |
    v
Append it to the text
    |
    v
Use the updated text to predict again
    |
    +---- repeat until finished
```

The model generates one character at a time because it uses a
character-level tokenizer.

## 11. Key settings

  Setting                            Value in the code
  -------------------------------- -------------------
  Batch size                                        32
  Context length                            256 tokens
  Embedding size                                   384
  Attention heads                                    6
  Transformer blocks                                 6
  Training iterations configured                 3,000
  Dropout                                          0.2
  Optimizer                                      AdamW

These are configured values, not a claim about the model's output
quality.

## 12. Project files

  File                    Purpose
  ----------------------- -------------------------------------------
  `final_gpt.py`          Main model, training, and generation code
  `final_gpt (1).ipynb`   Notebook version
  `input.txt`             Training text
  `README.md`             Project explanation

## 13. Run the project

``` bash
git clone https://github.com/SakshamBaurai/Mini_GPT.git
cd Mini_GPT
pip install torch
python final_gpt.py
```

**Note:** The script currently reads `/content/input.txt`, a Colab-style
path. For local execution, update it to the correct dataset path.

## 14. What I would improve next

-   [ ] Replace the character tokenizer with a BPE tokenizer.
-   [ ] Compare character-level and BPE tokenization on the same
    dataset.
-   [ ] Record validation loss and generated samples.
-   [ ] Add temperature and top-k sampling controls.
-   [ ] Save checkpoints so training can resume.
-   [ ] Make dataset paths work locally and in Colab.

For a fair comparison, keep the dataset and training setup as similar as
practical, and record the exact settings used.

## 15. Quick recall

  Concept                Remember it as
  ---------------------- ---------------------------------------
  Tokenizer              Translate text into IDs
  Embedding              Give each ID a learned representation
  Position               Tell the model where a token appears
  Attention              Weigh useful context
  Causal mask            No looking into the future
  Feed-forward network   Further process representations
  Loss                   Measure the error
  Backpropagation        Calculate gradients
  AdamW                  Update weights
  Generation             Predict, append, repeat

------------------------------------------------------------------------

**Final memory hook:** Translate → Represent → Connect → Block future →
Measure error → Calculate gradients → Update → Repeat.
