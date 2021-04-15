# UNIX only

FLAGS=

clean:
	rm -rf `find . -name __pycache__`
	find . -type f -name '*.py[co]'  -delete
	find . -type f -name '*~'  -delete
	find . -type f -name '.*~'  -delete
	find . -type f -name '@*'  -delete
	find . -type f -name '#*#'  -delete
	find . -type f -name '*.orig'  -delete
	find . -type f -name '*.rej'  -delete
	rm -f .coverage
	rm -rf coverage
	rm -rf build
	rm -rf htmlcov

