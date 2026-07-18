fetch-data:
	python -m src.data.eurostat

fetch-data-force:
	python -m src.data.eurostat --force

preprocess:
	python -m src.data.preprocessing

data:
	python -m src.data
