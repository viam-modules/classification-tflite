#!/bin/sh

python3 setup.py sdist --formats=gztar
pip install dist/model-0.1.tar.gz

python3 -m model.training
pip install pytest && pytest tests/