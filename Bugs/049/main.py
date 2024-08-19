import torch
import torch.nn as nn
import itertools
import sys
from tqdm import tqdm

class DBN(nn.Module):
    def fine_tuning(self, data, labels, num_epochs=10, max_iter=3):
        '''
        Parameters
        ----------
        data : TYPE torch.Tensor
            N x D tensor with N = num samples, D = num dimensions
        labels : TYPE torch.Tensor
            N x 1 vector of labels for each sample
        num_epochs : TYPE, optional
            DESCRIPTION. The default is 10.
        max_iter : TYPE, optional
            DESCRIPTION. The default is 3.

        Returns
        -------
        None.

        '''
        N = data.shape[0]
        #need to unroll the weights into a typical autoencoder structure
        #encode - code - decode
        for ii in range(len(self.rbm_layers)-1, -1, -1):
            self.rbm_layers.append(self.rbm_layers[ii])
        
        L = len(self.rbm_layers)
        optimizer = torch.optim.LBFGS(params=list(itertools.chain(*[list(self.rbm_layers[ii].parameters()) 
                                                                    for ii in range(L)]
                                                                  )),
                                      max_iter=max_iter,
                                      line_search_fn='strong_wolfe') 
        
        dataset     = torch.utils.data.TensorDataset(data, labels)
        dataloader  = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size*10, shuffle=True)
        #fine tune weights for num_epochs
        for epoch in range(1,num_epochs+1):
            with torch.no_grad():
                #get squared error before optimization
                v = self.pass_through_full(data)
                err = (1/N) * torch.sum(torch.pow(data-v.to("cpu"), 2))
            print("\nBefore epoch {}, train squared error: {:.4f}\n".format(epoch, err))
        
           #*******THIS IS THE PROBLEM SECTION*******#
            for ii,(batch,_) in tqdm(enumerate(dataloader), ascii=True, desc="DBN fine-tuning", file=sys.stdout):
                print("Fine-tuning epoch {}, batch {}".format(epoch, ii))
                with torch.no_grad():
                    batch = batch.view(len(batch) , self.rbm_layers[0].visible_units)
                    if self.use_gpu: #are we using a GPU?
                        batch = batch.to(self.device) #if so, send batch to GPU
                    B = batch.shape[0]
                    def closure():
                        optimizer.zero_grad()
                        output = self.pass_through_full(batch)
                        loss = nn.BCELoss(reduction='sum')(output, batch)/B
                        print("Batch {}, loss: {}\r".format(ii, loss))
                        loss.backward()
                        return loss
                    optimizer.step(closure)

data = torch.Tensor(10, 10)
labels = torch.Tensor(10, 1)
num_epochs = 10
max_iter = 3
dbnn = DBN()
# Run function
dbnn.fine_tuning(data, labels, num_epochs, max_iter)