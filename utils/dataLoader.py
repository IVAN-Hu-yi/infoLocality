from torch.utils.data import Dataset,DataLoader

class NgramsLanguageModelDataSet(Dataset):
    """
    Takes a corpus of text with one sentence per line and creates a dataset of n-grams
    """
    def __init__(self,N,data,tokenizer, left_padding=False):

        """
        formats the data into ngrams of size N, left padding with pad_ids if needed
        Args:
            N    (int): the size of the ngrams
            data (str): a string
            tokenizer : the tokenizer that splits the string into tokens
            left_padding (bool): if True, pads N-1 pad_ids to the left of each sentence
        """
        #performs various transformations of the data
        # self.data = [self.ngramify(tokenizer(sentence),N) for sentence in data]
        # self.data = [ngram for sent in self.data for ngram in sent if len(ngram) == N] #flattens a list of lists        
        self.N = N
        self.left_padding = left_padding
        self.tokenizer = tokenizer

        self.data = []
        tempIds = []
        for sentence in data:
            ids = self.tokenizer(sentence)
            if self.left_padding:
                ids = [self.tokenizer.pad_id]*(N-1) + ids
                tempIds.append(self.ngramify(ids,N))
            else:
                tempIds.append(self.ngramify(ids,N))

        for sent in tempIds:
            for ngram in sent:
                if len(ngram) == N:
                    self.data.append(ngram) #flattens a list of lists

            
    
    def ngramify(self,token_list,N):
        """
        Turns a list of tokens into a list of n-grams of size N
        The last token may have a size less than N
        """
        return [ token_list[idx:idx+N]  for idx in range(len(token_list)) ]
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

