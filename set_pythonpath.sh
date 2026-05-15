#! /bin/bash

export PYTHONPATH=$(git rev-parse --show-toplevel):$PYTHONPATH
echo $PYTHONPATH
