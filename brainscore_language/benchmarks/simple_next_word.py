import torch

class SimpleNextWord(): #TODO create abstract benchmark interface

    def __init__(self, benchmark_id):
        self.benchmark_if = benchmark_id

    def identifier(self):
        return self.benchmark_if

def predict_next_word(input, tokenizer, model):
    seq = input
    print("\nInput sequence: ")
    print(seq)

    inpts = tokenizer(seq, return_tensors="pt")
    print("\nTokenized input data structure: ")
    print(inpts)

    # res = model.generate(input_ids, **generator_args)
    # output = tokenizer.batch_decode(res, skip_special_tokens=True)

    # inpt_ids = inpts["input_ids"]  # just IDS, no attn mask
    # print("\nToken IDs and their words: ")
    # for id in inpt_ids[0]:
    #   word = tokenizer.decode(id)
    #   print(id, word)

    with torch.no_grad():
        output = model(**inpts)
    (loss, logits) = outputs[:2]

    print("\nAll logits for next word: ")
    print(logits)
    print(logits.shape)

    pred_id = torch.argmax(logits).item()
    print("\nPredicted token ID of next word: ")
    print(pred_id)

    pred_word = tokenizer.decode(pred_id)
    print("\nPredicted next word for sequence: ")
    print(pred_word)

    print("\nEnd demo ")
