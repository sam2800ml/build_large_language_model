
## Data sampling with slinding window
When we start to think in how we are going to pass to our  model the input text to be able to prodecit the next word, we need to introduce two new variables, one will be x and the other is y.
- x -> input tokens
- y -> target
- context_size -> this is the slinding window that we are going to be using

Imagine we have the following text -> "Hello world my name is santiago and i love programming"
if we chose a context_size of two it will look like this 
>
x = Hello world
y = world my
>
the model it will be the target token that the model is going to predict

we can also use a library as pytorch to be able to use tensors for a more efficient process, we are going to be changing the x for input_tensor and y for target_tensor.
>
The main parameters in loading the dataset is the following
- batch_size -> the batch size determine how many groups we have, of input and targets
- max_length -> how many samples we have in the testing
- stride -> this is like our slinding window, were we are going to take after the number of words, a small number is better but the computational cost is high

The process is being the following:
- First we split the text into just words separated
- Second we tokenized the single text
- Third we put ids in to the tokens
- fourth convert the token ids into embedding vectors

## creating token embeddings
We already have the tokens ids, with this we already have our dataset, but to be able to use it we have to convert it to embeddings vectors, before anything we can start defining the the size of the vocab and also the output dim, all these embeddings weights are going to be initialize with random values, this is because with the backpropagation those weights are going to be update it, to be able to increase the performance of the model.
- Token embedding:\
    in this process when we are embedding the vectors, if we have words that repeat these are going to have the same vector representation
    nowadays we have the self-attention mechanism which is a position agnostic, is helpfull to inject additional position information into the llm
    one of the option is create an embedding the same size as out input with a number position, each token has their own embedding
    