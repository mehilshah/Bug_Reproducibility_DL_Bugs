import torch

# here is a one-hot encoded vector for the multi-label classification
# the image thus has 2 correct labels out of a possible 3 classes
y = [0, 1, 1]

# these are some made up logits that might come from the network.
vec = torch.tensor([0.2, 0.9, 0.7])

def concurrent_softmax(vec, y):
    for i in range(len(vec)):
        zi = torch.exp(vec[i])
        sum_over_j = 0
        for j in range(len(y)):
            sum_over_j += (1-y[j])*torch.exp(vec[j])

        out = zi / (sum_over_j + zi)
        yield out

for result in concurrent_softmax(vec, y):
    print(result)