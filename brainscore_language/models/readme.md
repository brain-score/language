### How-to work with `HuggingfaceSubject`:
`HuggingfaceSubject` is a sub-class of the abstract class `ArtificialSubject` and allows experimenters to replicate human natural language experiments on models from the Hugging Face library of models.

First, the model is set-up as a `HuggingfaceSubject` instance. 

    model = HuggingfaceSubject(model_id='distilgpt2',
                                   region_layer_mapping={}
                                   )


Next, the model is instructed what kind of behavioral or neural recording task needs to be done. 

    model.perform_behavioral_task(task=ArtificialSubject.Task.reading_times) 
or 
    
    model.perform_neural_recording(recording_target=ArtificialSubject.RecordingTarget.language_system,
                                       recording_type=ArtificialSubject.RecordingType.spikerate_exact)

New `task` and `recording_type` are to be defined both in the `ArtificialSubject` class' `Task` class inside `artificial_subject.py`, and can be modularly accessed inside through `self.task_function_mapping_dict` in `HuggingfaceSubject`'s `__init__`.

Lastly, the task is performed using

        reading_times = model.digest_text(text)['behavior'].values
to get outputs of the task, or using 

        representations = model.digest_text(text)['neural']
to get the reprensetation behind the task. 

For more examples of usage of these models, refer to the unit tests in `tests/test_models/test_huggingface.py`. 
