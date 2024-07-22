#!/bin/sh

python3 -m model.training
pip install pytest && pytest