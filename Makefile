.PHONY: default init install doc test

default: init test install

init:
	pip install -r requirements.txt

install:
	pip install -e .
    
doc:
	cd docs && make singlehtml

test:
	python -m unittest

