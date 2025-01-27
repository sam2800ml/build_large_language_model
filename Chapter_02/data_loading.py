import torch 
from torch.utils.data import Dataset, DataLoader
import tiktoken
class GptDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]
    
def create_dataloader(txt, batch_size = 4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GptDataset(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers
    )
    return dataloader
with open("the-verdict.txt","r", encoding="utf-8") as f:
    raw_text = f.read()

dataloader = create_dataloader(txt=raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
data_iter = iter(dataloader)
input, target = next(data_iter)
print(f"input: {input}")
print(f"input shape: {input.shape}")

vocab_size_2 = 50257 # total of unique tokens
output_dim2 = 256 # size of the embedding
token_embedding_layer = torch.nn.Embedding(vocab_size_2, output_dim2)

toke_embedding = token_embedding_layer(input)
print(toke_embedding.shape)

context_length = 4
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim2)
pos_embedding = pos_embedding_layer(torch.arange(context_length))

print(pos_embedding)
print(f" context embedding: {pos_embedding.shape}")

input_embedding = toke_embedding + pos_embedding
print(input_embedding.shape)