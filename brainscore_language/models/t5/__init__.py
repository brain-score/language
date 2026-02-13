from brainscore_language import model_registry
from brainscore_language import ArtificialSubject
from brainscore_language.model_helpers.huggingface import HuggingfaceSubject
from transformers import AutoModelForSeq2SeqLM

# layer assignment based on choosing the maximally scoring layer on Pereira2018-encoding

# T5 models

model_registry['t5-small'] = lambda: HuggingfaceSubject(
    model_id='google/t5-v1_1-small', 
    model=AutoModelForSeq2SeqLM.from_pretrained('google/t5-v1_1-small', device_map="auto"),
    region_layer_mapping={ArtificialSubject.RecordingTarget.language_system: 'decoder.block.6'}
)

model_registry['t5-base'] = lambda: HuggingfaceSubject(
    model_id='google/t5-v1_1-base', 
    model=AutoModelForSeq2SeqLM.from_pretrained('google/t5-v1_1-base', device_map="auto"),
    region_layer_mapping={ArtificialSubject.RecordingTarget.language_system: 'encoder.block.9'}
)

model_registry['t5-large'] = lambda: HuggingfaceSubject(
    model_id='google/t5-v1_1-large', 
    model=AutoModelForSeq2SeqLM.from_pretrained('google/t5-v1_1-large', device_map="auto"),
    region_layer_mapping={ArtificialSubject.RecordingTarget.language_system: 'encoder.block.17'}
)

model_registry['t5-xl'] = lambda: HuggingfaceSubject(
    model_id='google/t5-v1_1-xl', 
    model=AutoModelForSeq2SeqLM.from_pretrained('google/t5-v1_1-xl', device_map="auto"),
    region_layer_mapping={ArtificialSubject.RecordingTarget.language_system: 'decoder.block.2'}
)

model_registry['t5-xxl'] = lambda: HuggingfaceSubject(
    model_id='google/t5-v1_1-xxl', 
    model=AutoModelForSeq2SeqLM.from_pretrained('google/t5-v1_1-xxl', device_map="auto"),
    region_layer_mapping={ArtificialSubject.RecordingTarget.language_system: 'decoder.block.0'}
)

# Flan-T5 models

model_registry['flan-t5-small'] = lambda: HuggingfaceSubject(
    model_id='google/flan-t5-small', 
    model=AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-small', device_map="auto"),
    region_layer_mapping={ArtificialSubject.RecordingTarget.language_system: 'encoder.block.7'}
)

model_registry['flan-t5-base'] = lambda: HuggingfaceSubject(
    model_id='google/flan-t5-base', 
    model=AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base', device_map="auto"),
    region_layer_mapping={ArtificialSubject.RecordingTarget.language_system: 'decoder.block.7'}
)

model_registry['flan-t5-large'] = lambda: HuggingfaceSubject(
    model_id='google/flan-t5-large', 
    model=AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-large', device_map="auto"),
    region_layer_mapping={ArtificialSubject.RecordingTarget.language_system: 'encoder.block.18'}
)

model_registry['flan-t5-xl'] = lambda: HuggingfaceSubject(
    model_id='google/flan-t5-xl', 
    model=AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-xl', device_map="auto"),
    region_layer_mapping={ArtificialSubject.RecordingTarget.language_system: 'decoder.block.2'}
)

model_registry['flan-t5-xxl'] = lambda: HuggingfaceSubject(
    model_id='google/flan-t5-xxl', 
    model=AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-xxl', device_map="auto"),
    region_layer_mapping={ArtificialSubject.RecordingTarget.language_system: 'decoder.block.0'}
)

# Flan-Alpaca T5 models

model_registry['flan-alpaca-base'] = lambda: HuggingfaceSubject(
    model_id='declare-lab/flan-alpaca-base', 
    model=AutoModelForSeq2SeqLM.from_pretrained('declare-lab/flan-alpaca-base', device_map="auto"),
    region_layer_mapping={ArtificialSubject.RecordingTarget.language_system: 'encoder.block.11'}
)

model_registry['flan-alpaca-large'] = lambda: HuggingfaceSubject(
    model_id='declare-lab/flan-alpaca-large', 
    model=AutoModelForSeq2SeqLM.from_pretrained('declare-lab/flan-alpaca-large', device_map="auto"),
    region_layer_mapping={ArtificialSubject.RecordingTarget.language_system: 'encoder.block.19'}
)

model_registry['flan-alpaca-xl'] = lambda: HuggingfaceSubject(
    model_id='declare-lab/flan-alpaca-xl', 
    model=AutoModelForSeq2SeqLM.from_pretrained('declare-lab/flan-alpaca-xl', device_map="auto"),
    region_layer_mapping={ArtificialSubject.RecordingTarget.language_system: 'decoder.block.5'}
)

model_registry['flan-alpaca-xxl'] = lambda: HuggingfaceSubject(
    model_id='declare-lab/flan-alpaca-xxl', 
    model=AutoModelForSeq2SeqLM.from_pretrained('declare-lab/flan-alpaca-xxl', device_map="auto"),
    region_layer_mapping={ArtificialSubject.RecordingTarget.language_system: 'decoder.block.0'}
)

# Other models based on T5

model_registry['flan-gpt4all-xl'] = lambda: HuggingfaceSubject(
    model_id='declare-lab/flan-gpt4all-xl', 
    model=AutoModelForSeq2SeqLM.from_pretrained('declare-lab/flan-gpt4all-xl', device_map="auto"),
    region_layer_mapping={ArtificialSubject.RecordingTarget.language_system: 'decoder.block.2'}
)

model_registry['flan-sharegpt-xl'] = lambda: HuggingfaceSubject(
    model_id='declare-lab/flan-sharegpt-xl', 
    model=AutoModelForSeq2SeqLM.from_pretrained('declare-lab/flan-sharegpt-xl', device_map="auto"),
    region_layer_mapping={ArtificialSubject.RecordingTarget.language_system: 'decoder.block.5'}
)

model_registry['flan-alpaca-gpt4-xl'] = lambda: HuggingfaceSubject(
    model_id='declare-lab/flan-alpaca-gpt4-xl', 
    model=AutoModelForSeq2SeqLM.from_pretrained('declare-lab/flan-alpaca-gpt4-xl', device_map="auto"),
    region_layer_mapping={ArtificialSubject.RecordingTarget.language_system: 'decoder.block.2'}
)
