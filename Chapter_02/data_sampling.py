import tiktoken

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

tokenizer = tiktoken.get_encoding("gpt2")
dataset_tokenize =  tokenizer.encode(raw_text)
print(len(dataset_tokenize))

context_size= 4
x = dataset_tokenize[:context_size]
y = dataset_tokenize[1:context_size+1]
print(x,y)

for i in range(1, context_size+1):
    context = dataset_tokenize[:i]
    desired = dataset_tokenize[i]
    print(f"Context= {tokenizer.decode(context)}, desired= {tokenizer.decode([desired])}")