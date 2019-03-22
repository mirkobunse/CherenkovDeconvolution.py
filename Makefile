default: init test install

init:
	python3 -m pip install -r requirements.txt

test:
	python3 -m unittest -v

install:
	python3 -m pip install -e .

doc:
	cd docs && make singlehtml

.PHONY: default init test install doc
