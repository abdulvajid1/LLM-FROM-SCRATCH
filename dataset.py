from torch.utils.data import DataLoader, Dataset
import torch
import tiktoken


class GPTDataset(Dataset):
    def __init__(self, text, tokenizer, max_len, stride):
        super().__init__()
        self.input_ids = []
        self.target_ids = []
        tokens = tokenizer.encode(text, allow_special=['<|endoftext|>'])
        for i in range(0, len(tokens) - max_len, stride):
            self.input_ids.append(torch.tensor(tokens[i: max_len]))
            self.target_ids.append(torch.tensor(tokens[i+1, i + max_len+1]))
            
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]
  
    
def create_dataloader(txt, batch_size=4, max_len=256,
                      stride=256, shuffle=True,
                      drop_last=True, num_workers=0):
    
    tokenizer = tiktoken.get_encoding('gpt2')
    dataset = GPTDataset(text=txt, tokenizer=tokenizer, max_len=max_len, stride=stride)
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        drop_last=True,
        num_workers=num_workers
    )
    return dataloader
    
    