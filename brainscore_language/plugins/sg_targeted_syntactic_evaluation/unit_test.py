# This is based on example Python API usage from the SyntaxGym website ( https://cpllab.github.io/syntaxgym-core/quickstart.html )
# To run this, you must have LM-Zoo and SyntaxGym installed.
from lm_zoo import get_registry
from syntaxgym import compute_surprisals, evaluate

# Retrieve model
model = get_registry()["huggingface://gpt2"] # Use this line to run huggingface models (DOES NOT REQUIRE DOCKER)
# model = get_registry()["gpt2"] # Use this line to run LM-Zoo models.  (REQUIRES DOCKER)

# Compute region-level surprisal data for our suite.
suite = compute_surprisals(model, "test_suite.json")

# Check predictions and return a dataframe containing a boolean value for each item (sentence pair) in test_suite.json.
results = evaluate(suite)
print(results)

# Calculate a "score"
print(sum(results["result"])/len(results["result"]))
#results2=results.to_csv(sep="\t")


