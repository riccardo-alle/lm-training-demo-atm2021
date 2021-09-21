from transformers import PreTrainedTokenizerFast

path = "/lm_demo/tokenization/tokenizertokenizer.json"
tokenizer = PreTrainedTokenizerFast(tokenizer_file=path)

print(tokenizer("this is the title of an Amazon product"))

