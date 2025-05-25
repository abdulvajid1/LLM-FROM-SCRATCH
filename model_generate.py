import torch


def generate(model,
             tokenizer,
             device='cpu',
             starting_context:str='i am a good',
             max_len=10,
             sampling=True,
             temperature=0.0,
             top_k=None,
             eos_id=None):
    
    model.eval()
    model.to(device);
    input_ids = tokenizer.encode(starting_context)
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    input_ids = input_ids.to(device)
    
    for i in range(max_len):
        with torch.no_grad():
            logits = model(input_ids)
            logits = logits[:,-1,:]
            
            if sampling:
                
                if top_k:
                    topk_logits, topk_pos = torch.topk(logits, k=top_k, dim=-1)
                    logits = torch.where(input=torch.tensor(float('-inf')),
                                         condition=logits < topk_logits[:,-1].reshape(-1, 1), 
                                         other=logits)
                if temperature>0.0:
                    logits = logits / temperature
                    
                probas = torch.nn.functional.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probas, num_samples=1)
                input_ids = torch.concat([input_ids, idx_next], dim=-1)
            else:
                assert temperature==0.0 and top_k is None, "You can't set temperature or topk if sampling=False"
                last_token = torch.argmax(logits, dim=-1).unsqueeze(0)
                input_ids = torch.cat([input_ids, last_token], dim=-1)
    return tokenizer.decode(input_ids.squeeze().tolist())