import ast
import sys

from brainscore_core.submission.endpoints import process_github_submission


if __name__ == '__main__':
	args_dict = ast.literal_eval(sys.argv[1])
	process_github_submission(args_dict)
