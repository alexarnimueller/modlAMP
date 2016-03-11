init:
	pip install -r requirements.txt

build:
	python setup.py build

install:
	python setup.py install

doc:
	open docs/build/html/READMEinclude.html
