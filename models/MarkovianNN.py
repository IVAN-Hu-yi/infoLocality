import torch.nn as nn
from torch.nn.functional import tanh,log_softmax
import torch

class NNLM (nn.Module):

    def __init__(self,emb_size,vocab_size,hidden_size,memory_size,pad_id):
      """
      A markovian Neural Network Language model 
      Args:
        emb_size (int)   : size of the embeddings (input and output embeddings have same size)
        vocab_size (int) : size of the vocabulary
        hidden_size (int): size of the hidden layer
        memory_size (int): size of the memory
        pad_id (int)     : id of the padding token
      """
      super().__init__()
      self.pad_id  = pad_id

      #Allocates the parameters
      self.wordemb = nn.Embedding(vocab_size,emb_size,padding_idx=pad_id)
      self.lm_in   = nn.Linear(emb_size*memory_size,hidden_size)
      self.lm_out  = nn.Linear(hidden_size,vocab_size)

    def forward(self,X):
        """
        This is the prediction function. It takes as input a batch of token codes
        and returns a batch of logits as output
        Args:
          X (torch.tensor) : a batch of token codes of size (Batch,Seq)
        Returns:
          (a batch of logits, a batch of embeddings). Both are torch.tensor of size (Batch,Seq,Emb)

        """
        input_embeddings      = self.wordemb(X)                              #broadcasting is used batchwise
        input_embeddings      = torch.flatten(input_embeddings,start_dim=1)  #concatenates the embeddings
        hidden_embeddings     = self.lm_in(input_embeddings)                 #broadcasting is used batchwise
        logits                = self.lm_out(tanh(hidden_embeddings))         #broadcasting is used batchwise
       
        #adding the softmax is useless here
        return logits

    
    def __call__(self,batch,device='cpu'):
        """
        This is the predict function
        An interface for the model to external calls. Outputs the logits that can be used as a starting point to compute surprisals
        Args:
            X (tensor): A batch of ngrams
        Returns:
            the logit (base e) for the last token in the ngram, that is log P(w_i | w_<i)
        """
        X      = batch[:,:-1] #drop the last word of the ngram from the context
        Y      = batch[:,-1] #last token of the ngrams
        logits = log_softmax(self.forward(X),dim=1)

        #last step: gather the logit for each reference  word
        return torch.gather(logits,1,Y.unsqueeze(0).T).squeeze()


    
    def train(self,dataloader,epochs,device='cpu'):
        """
        Trains a language model from scratch
        Args:
          dataloader (Dataloader): a torch Dataloader object serving the train set
          epochs (int): the number of epochs
          device (str): the device where to run the computations (cpu or cuda)
        """
        #Moves all the parameters to the compute device
        for p in self.parameters():
          p.to(device)

        cross_entropy = nn.CrossEntropyLoss(ignore_index=self.pad_id)
        optimizer     = torch.optim.AdamW(self.parameters(), lr=0.005)

        for epoch in range(epochs):
          loss_lst = []
          for batch in dataloader:
            X =  batch[:,:-1] #drop the last word of the ngram from the context
            Y =  batch[:,-1] #last word of the ngram

            X = X.to(device)
            Y = Y.to(device)
              
            #forward pass
            logits    = self.forward(X)
            loss      = cross_entropy(logits,Y)

            #backward pass
            loss.backward()
            loss_lst.append(loss.item())

            #update and reset
            optimizer.step()
            optimizer.zero_grad()
          # print(f'Epoch {epoch}, loss {sum(loss_lst)/len(loss_lst)}')
