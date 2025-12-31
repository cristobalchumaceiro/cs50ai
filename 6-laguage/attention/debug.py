from mask import *

text = input("Text: ")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
inputs = tokenizer(text, return_tensors="tf")
mask_token_index = get_mask_token_index(tokenizer.mask_token_id, inputs)

