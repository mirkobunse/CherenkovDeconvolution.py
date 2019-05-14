default: install

test:
	python3 setup.py test

install:
	python3 -m pip install -e .

doc:
	cd docs && make singlehtml

.PHONY: default init test install doc
