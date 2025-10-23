---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.6
  kernelspec:
    display_name: nlp
    language: python
    name: python3
---

# Dependency locality and information locality


This exercise aims to reproduce and evaluate the methodology related to assess dependency locality and information locality in natural languages. 

We expect you to work in binoms and return a document commenting your answers. This document will be supported by the code in the notebook.

The goal is to figure out how to carry out experiments like those described in the  paper *Modeling word and morpheme order in natural language as an efficient tradeoff of memory and surprisal* by Hahn, Degen and Futrell.

We will investigate how to design language models with limited memory and how to use them to answer questions on natural and manipulated datasets. The first part of the exercise is methodological. It amounts to design and set up non standard language models. The second part of the exercise aims to investigate how to use these models to answer scientific questions.  





**The answers to the exercises have to be returned as a written document (approx 5 pages). The code will be given in appendix. The grading uses mainly the text.** 


## Memory limited neural language models


We consider two types of language models, first Neural Network Language models (NNLM) then Recurrent Neural Network Language Models (RNNLM)
These are models predicting the $i-$th word given the $k$ previous ones. We provide an example implementation for an NNLM and you will have to design an implementation for the RNNLM in pytorch


### The tokenizer

The first step is to implement a tokenizer. The tokenizer plays two roles
 - it splits the input string into tokens
 - it maps the tokens to integer codes

We design a class that does exactly that. Note that we have to add some artificial tokens to handle unknown words for instance

The example tokenizer follows loosely the [HuggingFace tokenizer](https://huggingface.co/docs/transformers/en/main_classes/tokenizer) interface

```python
from utils.tokenizer import NaiveTokenizer
```

```python
# Toyish example
# Create the tokenizer with some known vocabulary

# observe how the unknown vocabulary is encoded
# observe how padding is encoded too

tokenizer = NaiveTokenizer("Language models are cool .".split())
codes     = tokenizer("Language models are not so cool .")
toks      = tokenizer.tokenize("Language models are not so cool .")

print("codes",codes)
print("toks",toks)

#Example of padding
print("\npadded codes")
batch = [tokenizer(sentence) for sentence in ["Language models are cool .","Language models are not so cool ."]]
print(tokenizer.pad_batch(batch))

```

### Dataset and Dataloader

[Dataset and Dataloader](https://pytorch.org/docs/stable/data.html) are pytorch classes that are used to load efficiently a dataset by batches on one or several GPUs. Here we provide a naive introductory example for language modeling


```python
from dataLoader import NgramsLanguageModelDataSet
from torch.utils.data import DataLoader
from utils.sentenceProcessing import normalize


#Dummy example to see what it does

zebra_dataset = """
There are five houses.
The Englishman lives in the red house.
The Spaniard owns the dog.
Coffee is drunk in the green house.
The Ukrainian drinks tea.
The green house is immediately to the right of the ivory house.
The Old Gold smoker owns snails.
Kools are smoked in the yellow house.
Milk is drunk in the middle house.
The Norwegian lives in the first house.
The man who smokes Chesterfields lives in the house next to the man with the fox.
Kools are smoked in the house next to the house where the horse is kept.
The Lucky Strike smoker drinks orange juice.
The Japanese smokes Parliaments.
The Norwegian lives next to the blue house.
"""
sentences = [normalize(sent) for sent in zebra_dataset.split('\n') if sent and not sent.isspace()]
tokenizer = NaiveTokenizer(normalize(zebra_dataset).split())
dataset   = NgramsLanguageModelDataSet(5,sentences,tokenizer) #pentagram dataset

print('DataLoader output\n')
dataloader = DataLoader(dataset, batch_size=10, shuffle=True,collate_fn=tokenizer.pad_batch)
for batch in dataloader:
  print(batch)
  size = batch.size()
  print("batch size:",size)
```

```python
import torch
from models.MarkovianNN import NNLM
```

```python
lang_model = NNLM(128,tokenizer.vocab_size,128,5-1,tokenizer.pad_id)
lang_model.train(dataloader,40,device='cpu') #or 'cuda' to run on a GPU or 'mps' for latest macos

#predicting with the model

test_set    = NgramsLanguageModelDataSet(5,["The Lucky Strike smoker drinks orange juice ."],tokenizer)
test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False,collate_fn=tokenizer.pad_batch)

for batch in test_loader:
    predictions = lang_model(batch)
    src    = tokenizer.decode_ngram(batch)
    print(src)
    logits = lang_model(batch)
    print(list(zip(src.split(),logits.tolist())))
   
    
```

**Exercise 1 (1pt)** Read the code above and try to understand it.  Why do we see only the last few tokens of the test sentence ? This is a common issue in many language modeling implementations. Provide a correction to the code and explain what was the cause. 


**Exercise 2 (1pt)** Try to understand how the language model  handles unknown words. Is it the same method as GPT-2 ? explain the differences.
Can we easily compare the suprisals returned by the two models ? what is your opinion?


**Exercise 3 (3pts)** Given the code above, implement a Recurrent Neural Network Language Model trainable on ngrams. The dataset and the dataloader can be reused as is. The main difference is located in the Module subclass. This time we do not concatenate the embeddings of the past tokens but rather the model uses an RNN module. Pytorch library provides an LSTM module and there are [several tutorials explaining how to implement those models online](https://docs.pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html).  

Explain what are the main challenges related to implementing LSTM like models in pytorch. Do you manage to run your models on cuda devices ?


## Generating artificial datasets


In order to test to which extent human language minimize dependency length and enforce information locality,
we will compare word order in natural corpus with word order in corpora where word order is somewhat randomized.
The corpora used here are taken from Universal Dependencies.



**Exercise 4 (1pt)** Download the UD corpora from the website and choose 10 languages with which you will work. Some of them should be free word order some 
of them should be fixed word order. Ideally corpora should be large enough to estimate language models. To figure out whether a language is free or fixed word order you may look at resources like the [World Atlas of Language Structures](https://wals.info)
Write an explanation on how you chose your corpora.


**Exercise 5 (4pt)** Generate artificial corpora from the natural corpora. Those artificial corpora will have randomized word order. 
The `dependency.py` module given in the git may help you read the corpora and shuffle the dependency trees ? 
As controlling randomness is a key issue you may wish to consult the paper *Modeling word and morpheme order in natural language as an efficient tradeoff of memory and surprisal* 
and decide if the default randomization procedures suit your needs.


## Dependency Length experiments


**Exercise 6 (5pts)** Try to answer the following questions. Is there a significative difference between dependency length observed in natural corpora with the dependency length observed in corpora with a randomized word order ? Illustrate the differences with relevant plots. Can you identify an hypothesis test that would allow to test the observed differences ? Do languages said to be free word order have indeed longer average dependency lengths ?  How important is the choice of the randomization procedure to reach your conclusion ? You may combine observations from plots and also relevant hypothesis testing to support your answers.


## Information Locality experiments


**Exercise 7(4 pts)** Mirroring the previous exercise, we aim to assess for locality effects in natural language without measuring dependency lengths but rather using information theoretic measures computed from language models. To test the relevance of the information locality theorem, train language models with different memory sizes (NNLM, RNNLM or others) and for each of your corpora, compare the average surprisals of the models as a function of the memory size. What do you observe ? how do you interpret the results ? How would you compare the results across languages with different word order properties? is it straightforward ? explain the problematics and illustrate your conclusions with plots and/or hypothesis tests.


**Exercise 8 (1pt)** It would be tempting to use a multilingual large language model here. Is it possible ? conjecture whether they could be used or not ? what are the main challenges ?
