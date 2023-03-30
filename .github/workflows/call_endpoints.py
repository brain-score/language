import ast
import sys

from brainscore_core.submission.endpoints import process_github_submission, get_email_from_uid


if __name__ == '__main__':
	function = getattr(sys.modules[__name__], sys.argv[1])
	args = ast.literal_eval(sys.argv[2])
	function(args)