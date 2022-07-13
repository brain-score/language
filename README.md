[![Build Status](https://app.travis-ci.com/brain-score/language.svg?token=vqt7d2yhhpLGwHsiTZvT&branch=main)](https://app.travis-ci.com/brain-score/language)
[![Documentation Status](https://readthedocs.org/projects/brain-score_language/badge/?version=latest)](https://brain-score_language.readthedocs.io/en/latest/?badge=latest)
[![Website Status](https://img.shields.io/website.svg?down_color=red&down_message=offline&up_message=online&url=http%3A%2F%2Fbrain-score.org/language)](https://brain-score.org/language)

Brain-Score Language is a platform to evaluate computational models of language on their match to behavioral and neural
measurements in the domain of language processing. The intent of Brain-Score is to adopt many (ideally all) the
experimental benchmarks in the field for the purpose of model testing, falsification, and comparison. To that end,
Brain-Score operationalizes experimental data into quantitative benchmarks that any model candidate following
the `BrainModel` interface can be scored on.

[//]: # (See the [Documentation]&#40;https://brain-score_language.readthedocs.io&#41; for more details.)

#Installation requirements:
`git clone` the following repos:
- https://github.com/brain-score/core
- https://github.com/brain-score/result_caching
- https://github.com/brain-score/brainio

Once this is done, make sure to move the top level directory of the same name into the directory in this folder. E.g. This is what your directory should look like:
```
|
|-brainio
|-brainscore_core
|-brainscore_language
|-result_caching
|-tests
|-LICENSE
|-pyproject.toml
|-README.md
|-run_test_coverage.sh
```


Brain-Score is made by and for the community. To contribute,
please [send in a pull request](https://github.com/brain-score/language/pulls).

## License

MIT license
