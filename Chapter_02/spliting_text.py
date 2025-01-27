import re 


file_path = "the-verdict.txt"

with open(file_path, 'r') as file:
    file_contents = file.read()

result = re.split(r'([,.:;?_!"()\']|--|\s)', file_contents)
result = [item.strip() for item in result if item.strip()]
print(result[:30])

all_words = sorted(set(result))
all_words.extend(["<|endoftext|>","<|unk|>"])
print(len(all_words))
print(all_words[:30])

#creating the vocabulary
vocab = {token:integer for integer, token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(item)


class simpleTokenizerV1:
    def __init__(self, vocab):
        self.input_to_token = vocab
        self.token_to_input = {i:s for s,i in vocab.items()}

    def encode(self, text):
        result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        result = [item.strip() for item in result if item.strip()]
        result = [item if item in self.input_to_token else "<|unk|>" for item in result]
        ids = [self.input_to_token[s] for s in result]
        return ids
    
    def decode(self, ids):
        text = " ".join([self.token_to_input[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
    

tokenizer = simpleTokenizerV1(vocab)
text = """ Hello Chicago, Dubarry! pardonable pride"""
ids = tokenizer.encode(text)
print(ids)
print(tokenizer.decode(ids))



