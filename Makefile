.PHONY: default init test install doc

default: init test install

init:
	pip install -r requirements.txt

test:
	python -m unittest

install:
	pip install -e .
    
doc:
	cd docs && make singlehtml

