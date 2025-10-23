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
