
import langbrainscore 
from langbrainscore import encoding






def main():
    
    pereira18_encoder = encoding.brain.Pereira18Encoder(data_dir = )
    distilgpt2 = encoding.silico.HuggingfaceEncoder(model_name_or_path = 'distilgpt2')

    bs = langbrainscore.score(pereira18_encoder, distilgpt2)


if __name__ == '__main__':

    main()
