import torch

class NaiveTokenizer:
  """
  NaiveTokenizer that approximates the HuggingFace tokenizer interface
  """
  def __init__(self, base_vocabulary, unk='<unk>',pad='<pad>'):
      """
      Creates a tokenizer with some vocabulary and an unknown token
      Args:
        base_vocabulary: list of strings
        unk: string for unknown tokens
        pad: string for padding tokens
      """
      assert(type(base_vocabulary) == list)
      self.unk = unk
      self.pad = pad
      self.vocabulary = []
      self.types2idx  = {}
      self.add_tokens([self.unk,self.pad] + base_vocabulary)


  def add_tokens(self, tokens):
    """
    Adds a list of tokens to the vocabulary.
    Args:
      tokens :  a list of strings to add to the vocabulary
    """
    if not type(tokens) == list:
      tokens = [tokens]

    for token in tokens:
      if token not in self.vocabulary:
        self.vocabulary.append(token)

    self.types2idx = {elt:idx for idx,elt in enumerate(self.vocabulary)}

  def tokenize(self, string):
    """
    Splits a string into tokens
    Args:
      string : a string to tokenize
    Returns:
      a list of strings
    """
    tokens = string.split()
    return tokens

  def convert_tokens_to_ids(self, tokens):
    """
    Maps a list of tokens to integer codes
    Args:
      tokens : a list of strings
    Returns:
      a list of integers
    """
    unkid = self.types2idx[self.unk]
    return [self.types2idx.get(token,unkid) for token in tokens]

  def encode(self, string):
    """
    Encodes a string into a list of integers
     Args:
      string : a text to encode
    Returns:
      a list of integers
    """
    tokens = self.tokenize(string)
    return self.convert_tokens_to_ids(tokens)

  def decode(self,ids):
    """
    Decodes a list of integers into a string
    Args:
      ids : a list of integers
    Returns:
      a string
    """
    tokens = [self.vocabulary[idx] for idx in ids]
    return ' '.join(tokens)

  def __call__(self, string):
    """
    @see the encode method
    """
    return self.encode(string)

  @property
  def pad_id(self):
    """
    Returns the id of the pad token
    """
    return self.types2idx[self.pad]

  @property
  def vocab_size(self):
    """
    Returns the size of the vocabulary
    """
    return len(self.vocabulary)


  def decode_ngram(self,ngram_sequence):
    """
    Extracts a readable string from a sequence of ngrams
    """
    return self.decode(ngram[-1]  for ngram in ngram_sequence)
            

    
  def pad_batch(self,batch_codes):
    """
    Pads a batch of integers with the pad code
    Args:
      batch_codes : a list of lists of integers
    Returns:
      a list of lists of integers
    """
    max_len      = max([len(sentence) for sentence in batch_codes])
    padded_codes = [ sentence + [self.pad_id]*(max_len-len(sentence)) for sentence in batch_codes]
    return torch.LongTensor(padded_codes)
