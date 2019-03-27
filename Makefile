default: init test install

init:
	python3 -m pip install -r requirements.txt

test:
	python3 -m unittest -v tests/test_*.py

test-julia:
	python3 -m unittest -v tests/jl/test_*.py

install:
	python3 -m pip install -e .

doc:
	cd docs && make singlehtml

.PHONY: default init test test-julia install doc
