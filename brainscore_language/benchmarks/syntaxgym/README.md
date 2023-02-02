SyntaxGym is a unified platform for targeted syntactic evaluation of language models. It supports all steps
of the evaluation process, from designing test suites to visualizing final results. The goal of SyntaxGym 
is to make psycholinguistic assessment of language models more standardized, reproducible, and accessible 
to a wide variety of researchers.  See the [SyntaxGym Website](https://syntaxgym.org/) for more details.

## Benchmark: SyntaxGym (Center Embedding)
---
```yaml
benchmark details:
  name: SyntaxGym (Center Embedding)
  developer: MIT Computational Psycholinguistics Lab
  date: 2020
  version: 1.0
  type: behavioral
  description: |
    Center embedding, the ability to embed a phrase in the middle of another phrase of the same type, is a hallmark 
    feature of natural language syntax. Center-embedding creates nested dependencies, which could pose a challenge for 
    some language models. To succeed in generating expectations about how sentences will continue in the context of 
    multiple center embedding, a model must maintain a representation not only of what words appear in the preceding 
    context but also of the order of those words, and must predict that upcoming words occur in the appropriate order. 
    In this test suite we use verb transitivity and subject/verb plausibility to test model capabilities in this respect.

  license: MIT
  questions: mitcpllab@gmail.com
  citations: |
    @inproceedings{hu-etal-2020-systematic,
    title = "A Systematic Assessment of Syntactic Generalization in Neural Language Models",
    author = "Hu, Jennifer and Gauthier, Jon and Qian, Peng and Wilcox, Ethan and Levy, Roger",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics", 
    year = "2020", url = "https://www.aclweb.org/anthology/2020.acl-main.158", pages = "1725--1744"}
    @inproceedings{gauthier-etal-2020-syntaxgym, 
    title = "{S}yntax{G}ym: An Online Platform for Targeted Evaluation of Language Models", 
    author = "Gauthier, Jon and Hu, Jennifer and Wilcox, Ethan and Qian, Peng and Levy, Roger",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: System Demonstrations",
    year = "2020", url = "https://www.aclweb.org/anthology/2020.acl-demos.10", pages = "70--76"}

experiment:
  task: "reading_times"
  recording: NA
  experiment_card: NA
  bidirectionality: NA
  contextualization: NA

data:
  accessibility: public
  measurement_type: NA
  granularity: per sentence surprisal evaluation
  method: NA
  references: Wilcox E. Levy R. & Futrell R. (2019). 
  location: in the benchmark as a json

metric:
  mapping: NA 
  metric: accuracy
  error_estimation: cross validation over sentences, stratified by passage

ethical_considerations: NA

recommendations: NA

example_usage:
  example: see test_integration.py for example usage
  note: |
    To use your own suite specified as a [json file:](https://cpllab.github.io/syntaxgym-core/suite_json.html) 
    ```
    from brainscore_language.benchmarks.syntaxgym import SyntaxGymSingleTSE
    from brainscore_language import load_model
    model = load_model("distilgpt2")
    benchmark = SyntaxGymSingleTSE("path/to/my/syntaxgym/suite.json")
    score = benchmark(model)
    
    or
    
    from brainscore_language import load_model, load_benchmark
    model = load_model("distilgpt2")
    benchmark = load_benchmark("center_embed")
    score=benchmark(model)
    print(score)
    ```

