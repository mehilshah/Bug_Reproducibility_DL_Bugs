import torch
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
cls_token_id = tokenizer.cls_token_id
sep_token_id = tokenizer.sep_token_id
pad_token_id = tokenizer.pad_token_id

model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
model.eval()

def text_to_input(text):
  x = tokenizer.encode(text, add_special_tokens=False) # returns python list
  x = [cls_token_id] + x + [sep_token_id]
  token_count = len(x)
  pad_count = 512 - token_count
  x = x + [pad_token_id for i in range(pad_count)]
  return torch.tensor([x])

extract_embeddings = torch.nn.Sequential(list(model.children())[0])
rest_of_bert = torch.nn.Sequential(*list(model.children())[1:])

input_ids = text_to_input('A sentence.')
x_embedding = extract_embeddings(input_ids)
output = rest_of_bert(x_embedding)