from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

context_1 = 'here is some context_1 and some more stuff'
context_2 = 'here is some context and some more stuff and more stuff aspodkaspd'
answer_1 = 'this is not the answer'

input_ids_wrong = tokenizer(context_1 + answer_1, return_tensors="pt").input_ids
input_ids_correct = tokenizer(context_2 + answer_1, return_tensors="pt").input_ids
context_1_tokens_length = len(tokenizer(context_1, return_tensors="pt").input_ids[0])
context_2_tokens_length = len(tokenizer(context_2, return_tensors="pt").input_ids[0])

target_ids_wrong = input_ids_wrong.clone()
target_ids_correct = input_ids_correct.clone()

target_ids_wrong[:, :context_1_tokens_length] = -100 
target_ids_correct[:, :context_2_tokens_length] = -100 

print('target_ids_wrong', target_ids_wrong)
print('target_ids_correct', target_ids_correct)

with torch.no_grad():
    outputs_wrong = model(input_ids_wrong, labels=target_ids_wrong)
    outputs_correct = model(input_ids_correct, labels=target_ids_correct)
    
    neg_log_likelihood_wrong = outputs_wrong.loss
    neg_log_likelihood_correct = outputs_correct.loss

    ppl_wrong = torch.exp(neg_log_likelihood_wrong)
    ppl_correct = torch.exp(neg_log_likelihood_correct)
    print('ppl_wrong', ppl_wrong)
    print('ppl_correct', ppl_correct)