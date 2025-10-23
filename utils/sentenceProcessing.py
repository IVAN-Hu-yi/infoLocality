
def normalize(sentence):
    """
    Separates punctuation from words prior to tokenization
    """
    return sentence.replace('.', ' . ')