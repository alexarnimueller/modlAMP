init:
	pip install -r requirements.txt --user
	python setup.py build

install:
	python setup.py install

doc:
	open docs/build/html/index.html