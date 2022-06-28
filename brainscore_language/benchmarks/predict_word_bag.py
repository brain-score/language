def predict_next_word(input, tokenizer, model):
    """
    :param seq: the text to be used for inference e.g. "the quick brown fox"
    :param tokenizer: huggingface tokenizer, defined in the HuggingfaceModel class via: self.tokenizer =
    AutoTokenizer.from_pretrained(self.model_id)
    :param model: huggingface model, defined in the HuggingfaceModel class via: self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
    :return: single string which reprensets the model's prediction of the next word
    """
    import torch

    tokenized_inputs = tokenizer(input, return_tensors="pt")
    output = model(**tokenized_inputs, output_hidden_states=True, return_dict=True)
    logits = output['logits']
    pred_id = torch.argmax(logits, axis=2).squeeze()
    last_model_token_inference = pred_id[-1].tolist()
    next_word = tokenizer.decode(last_model_token_inference)

    return next_word, output['hidden_states']
