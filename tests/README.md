# Unit Tests

## Markers
Unit tests have various markers that denote possible issues in the travis build:

* **memory_intense**: tests requiring more memory than is available in the travis sandbox (currently 3 GB, https://docs.travis-ci.com/user/common-build-problems/#my-build-script-is-killed-without-any-error)
* **slow**: tests leading to runtimes that are not possible on the openmind cluster (>1 hour per test) 
* **travis_slow**: tests running for more than 10 minutes without output (which leads travis to error)

Use the following syntax to mark a test:
```
@pytest.mark.memory_intense
def test_something(...):
    assert False
```

To skip a specific marker, run e.g. `pytest -m "not memory_intense"`.
To skip multiple markers, run e.g. `pytest -m "not private_access and not memory_intense"`.
