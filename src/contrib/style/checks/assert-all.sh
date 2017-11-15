#!/bin/bash

# Checks for compliance with 
# Rule: 'Don't use assert().'


# Steps:
# - exclude src/docs/ holding the documentation only
# - Call checker script


find src/ include/ -name *.[ch] \
 | grep -v 'src/docs' \
 | xargs $(dirname $0)/assert.sh

