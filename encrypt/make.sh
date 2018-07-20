#!/usr/bin/env bash

export nuitka3=/Library/Frameworks/Python.framework/Versions/3.5/bin/nuitka
$nuitka3 --module encrypt/src/serialize.py --output-dir=encrypt/module --no-pyi-file --full-compat --remove-output
#$nuitka3 --module encrypt/src/timer.py --output-dir=encrypt/module
#$nuitka3 --module /Users/kyoka/Documents/coding/pycharm_workplace/python3learning/encrypt/pkg \
#--recurse-plugins=/Users/kyoka/Documents/coding/pycharm_workplace/python3learning/encrypt/pkg \
#--output-dir=encrypt/module --no-pyi-file --full-compat --remove-output