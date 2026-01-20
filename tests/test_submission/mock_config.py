from brainscore_language.submission import config
test_database = 'brainscore-ohio-test_language'
config.get_database_secret = lambda: test_database
