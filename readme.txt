# Mini GPT

This project is a small GPT style language model built from scratch using PyTorch.

The main purpose of this project is to understand what actually happens inside a GPT model when it reads text, learns patterns, and generates new text. Instead of using an existing language model, I built the main parts of the Transformer architecture myself so I could understand how everything works together.

This is mainly a learning project, but it also gives a practical look at the basic ideas behind modern language models.

## Project Files

The project currently contains the following files.

`final_gpt.py`

This is the main Python file containing the GPT implementation, training process, and text generation.

`final_gpt (1).ipynb`

This is the Jupyter Notebook version of the project. It makes it easier to go through the implementation and experiment with different parts of the model.

`input.txt`

This is the text dataset used for training the model.

`README.md`

This file explains the project and how the complete process works.

## How the Model Works

The basic idea is pretty simple.

The model takes text as input, converts that text into numbers, processes those numbers through the Transformer, learns from its mistakes, and eventually uses what it has learned to generate new text.

The complete process looks like this.

```text
input.txt
    ↓
Read the text
    ↓
Convert characters into numbers
    ↓
Create training and validation data
    ↓
Create batches of text
    ↓
Convert tokens into embeddings
    ↓
Add positional information
    ↓
Pass through Transformer blocks
    ↓
Self Attention
    ↓
Feed Forward Network
    ↓
Layer Normalization
    ↓
Generate predictions
    ↓
Calculate Loss
    ↓
Backpropagation
    ↓
Update Model Parameters
    ↓
Repeat the process
    ↓
Generate new text
```

## 1. Reading the Text

The first step is reading the contents of `input.txt`.

The model does not directly understand words or sentences. At this stage, it simply receives a large amount of text that it can learn patterns from.

For example, if the dataset contains something like this:

```text
hello world
hello there
hello everyone
```

the model will start learning which characters usually appear after other characters.

## 2. Converting Text Into Numbers

Computers cannot directly feed characters into a neural network, so the text needs to be converted into numbers.

This project uses character level tokenization.

Each unique character in the dataset gets its own number.

For example:

```text
h → 17
e → 5
l → 12
o → 15
```

So the word:

```text
hello
```

can be represented as:

```text
17 5 12 12 15
```

The model works with these numbers instead of the original characters.

There are two mappings used for this.

```text
Character → Number
Number → Character
```

The first one is used when preparing the training data.

The second one is used when converting the generated numbers back into readable text.

## 3. Splitting the Dataset

After converting the complete text into numbers, the dataset is divided into training data and validation data.

The training data is what the model learns from.

The validation data is used to check how well the model performs on text that it has not directly trained on.

This gives us a better idea of whether the model is actually learning useful patterns.

## 4. Creating Training Batches

The model does not process the entire dataset at once.

Instead, smaller pieces of the dataset are selected and processed in batches.

The model is trained to predict the next character based on the characters that came before it.

For example:

```text
Input:  h
Target: e

Input:  he
Target: l

Input:  hel
Target: l

Input:  hell
Target: o
```

This is the basic idea behind next token prediction.

The model keeps learning the relationship between what it has already seen and what should come next.

## 5. Token Embeddings

The numbers representing characters do not have much meaning by themselves.

For this reason, every token is converted into a vector using an embedding layer.

For example, instead of simply using:

```text
17
```

the model might represent it internally as something like:

```text
[0.21, -0.43, 0.72, 0.11, ...]
```

These vectors are learned during training.

As the model trains, it gradually learns useful representations for different characters and their relationships.

## 6. Positional Information

The Transformer also needs to know where each character appears in the sequence.

For example:

```text
I love AI
```

and:

```text
AI love I
```

contain similar characters, but the order is completely different.

The model therefore adds positional information to the token embeddings.

This allows it to understand that the first character, second character, third character, and so on occur at different positions.

## 7. Transformer Blocks

The embeddings are then passed through the Transformer blocks.

This is where most of the interesting processing happens.

Each Transformer block contains attention mechanisms and a feed forward network.

The information moves through the network while residual connections and layer normalization help keep the training process stable.

A simplified version looks like this.

```text
Input
  ↓
Self Attention
  ↓
Layer Normalization
  ↓
Feed Forward Network
  ↓
Layer Normalization
  ↓
Output
```

Multiple blocks can be stacked together to make the model more capable.

## 8. Self Attention

Self attention is one of the most important ideas in GPT.

It allows the model to look at the previous characters in the sequence and decide which ones are important when making its next prediction.

For example, consider:

```text
The dog was playing because it was happy.
```

When processing a later part of the sentence, the model can use information from earlier parts of the sentence to make a better prediction.

Attention uses three main components called Query, Key, and Value.

The basic calculation is:

```text
Attention(Q, K, V)
=
softmax(QKᵀ / √dₖ)V
```

The important idea is that the model calculates how much attention each token should give to other tokens.

## 9. Causal Attention

Since GPT generates text from left to right, the model should not be allowed to look at future characters while making a prediction.

For example, when predicting the next character, the model can look at everything that came before it, but it cannot look at the actual answer that comes later.

The attention mask makes sure this happens.

It creates a structure similar to this.

```text
        1   2   3   4
    1   ✓   ✗   ✗   ✗
    2   ✓   ✓   ✗   ✗
    3   ✓   ✓   ✓   ✗
    4   ✓   ✓   ✓   ✓
```

This keeps the model focused on the information that would actually be available when generating text.

## 10. Multi Head Attention

Instead of having only one attention mechanism, the model can use multiple attention heads.

Each head can focus on different relationships within the text.

The outputs from all the heads are then combined before being passed to the next part of the Transformer.

The idea can be visualized like this.

```text
                 Input
                   ↓
        ┌──────────┼──────────┐
        ↓          ↓          ↓
     Head 1      Head 2      Head 3
        ↓          ↓          ↓
        └──────────┼──────────┘
                   ↓
                Combine
                   ↓
                Output
```

This gives the model multiple ways to look at the same sequence.

## 11. Feed Forward Network

After attention, the information passes through a feed forward network.

This is another neural network inside the Transformer that transforms the information further.

A simplified version looks like this.

```text
Input
  ↓
Linear Layer
  ↓
Activation Function
  ↓
Linear Layer
  ↓
Output
```

The attention mechanism helps the model understand relationships between tokens, while the feed forward network processes and transforms those representations further.

## 12. Layer Normalization

Layer normalization is used to keep the values inside the network more stable during training.

It helps the different parts of the Transformer work together more smoothly and makes the learning process more stable.

## 13. Generating Predictions

After the text has passed through the Transformer blocks, the model produces a set of values called logits.

These values represent how likely each possible next character is.

For example, the model might produce something like:

```text
a → 1.2
b → 0.4
c → 2.8
d → 0.7
```

The model can then convert these values into probabilities.

For example:

```text
a → 0.10
b → 0.05
c → 0.75
d → 0.10
```

In this example, the model thinks that `c` is the most likely next character.

## 14. Calculating the Loss

During training, the model's prediction is compared with the actual next character from the dataset.

If the model predicts the wrong character, the loss will be higher.

If the prediction is closer to the correct answer, the loss will be lower.

Cross entropy loss is used to measure this difference.

The basic process is:

```text
Model Prediction
       ↓
Compare With Actual Character
       ↓
Calculate Loss
```

The goal of training is to reduce this loss over time.

## 15. Backpropagation

Once the loss has been calculated, the model needs to figure out how its parameters should change.

This is where backpropagation comes in.

The gradients calculated during backpropagation tell the model which parameters contributed to the error and how they should be adjusted.

The process looks like this.

```text
Prediction
    ↓
Loss
    ↓
Backpropagation
    ↓
Calculate Gradients
    ↓
Update Parameters
```

This process happens again and again during training.

## 16. Updating the Model

An optimizer such as AdamW is used to update the model parameters.

Every training step slightly changes the parameters so that the model can make better predictions in the future.

After many iterations, the model gradually becomes better at predicting the next character.

## 17. Generating Text

Once the model has been trained, it can be used to generate new text.

The generation process starts with some initial text.

For example:

```text
The
```

The model predicts the next character.

Maybe it predicts:

```text
c
```

Now the sequence becomes:

```text
The c
```

The model uses this new sequence to predict another character.

Maybe:

```text
a
```

Now it becomes:

```text
The ca
```

This process continues one character at a time.

```text
Starting Text
      ↓
Predict Next Character
      ↓
Add Character
      ↓
Use Updated Text
      ↓
Predict Again
      ↓
Repeat
```

This is how the model generates an entire piece of text.

## The Complete Picture

The complete process can be thought of as one big loop.

```text
Text Dataset
     ↓
Tokenization
     ↓
Numerical Representation
     ↓
Training Batches
     ↓
Token Embeddings
     ↓
Positional Information
     ↓
Transformer
     ↓
Self Attention
     ↓
Feed Forward Network
     ↓
Prediction
     ↓
Loss Calculation
     ↓
Backpropagation
     ↓
Parameter Updates
     ↓
Repeat
     ↓
Trained Model
     ↓
Text Generation
```

The most important thing I wanted to understand while building this project was what happens between the input text and the final generated text.

The model starts with something as simple as characters from a text file. Those characters are converted into numbers, turned into embeddings, processed through attention and neural network layers, and eventually turned back into predicted characters.

By repeating this process thousands of times, the model learns patterns from the training data and becomes capable of generating text that follows those patterns.

## Technologies Used

Python

PyTorch

Jupyter Notebook

Transformer architecture

Self Attention

Multi Head Attention

Character Level Tokenization

Neural Networks

Backpropagation

AdamW Optimizer

## Running the Project

Clone the repository.

```bash
git clone https://github.com/SakshamBaurai/Mini_GPT.git
```

Move into the project directory.

```bash
cd Mini_GPT
```

Install PyTorch if it is not already installed.

```bash
pip install torch
```

Run the Python implementation.

```bash
python final_gpt.py
```

You can also open `final_gpt (1).ipynb` in Jupyter Notebook or VS Code if you want to go through the implementation interactively.

## Final Note

This project is intentionally small compared to modern large language models.

The goal here is not to build something at the scale of ChatGPT. The goal is to understand the core ideas that make GPT models work.

Building this from scratch helped me understand how text gets converted into numbers, how attention processes those numbers, how the model learns from its mistakes, and how the same process can eventually be used to generate completely new text.

There is still a lot that can be improved, such as using a better tokenizer, a larger dataset, more Transformer layers, GPU training, better sampling methods, and longer training.

For now, this project serves as a practical starting point for understanding how GPT works internally.
