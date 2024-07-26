#!/bin/sh

python3 setup.py sdist --formats=gztar
pip install training_code/dist/model-0.1.tar.gz

python3 -m model.training --dataset_file=dataset.jsonl --model_output_directory=.
pip install pytest && pytest tests/
