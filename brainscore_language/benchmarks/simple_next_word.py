import torch

class SimpleNextWord(): #TODO create abstract benchmark interface

    def __init__(self, benchmark_id):
        self.benchmark_if = benchmark_id

    def identifier(self):
        return self.benchmark_if

def predict_next_word(input, tokenizer, model, **generator_args):
    seq = input
    print("\nInput sequence: ")
    print(seq)

    input_ids = tokenizer(seq, return_tensors="pt")

    res = model.generate(input_ids, **generator_args)

    output = tokenizer.batch_decode(res, skip_special_tokens=True)
    print(output)
