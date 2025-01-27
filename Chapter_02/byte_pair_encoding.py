import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

text = "Hello, my name is santiago, im 24 years old learning about life, sawasalo"
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)
strings = tokenizer.decode(integers)
print(strings)

# Exercise 2.1 Byte pair encoding of unknown words
unknown_word = "Akwirw ier"
uknnown_integer = tokenizer.encode(unknown_word)
print(uknnown_integer)
unknown_string = tokenizer.decode(uknnown_integer)
print(unknown_string)