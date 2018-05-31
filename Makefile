.PHONY: default init doc test

default: init test

init:
	pip install -r requirements.txt

doc:
	cd docs && make singlehtml

test:
	python -m unittest

