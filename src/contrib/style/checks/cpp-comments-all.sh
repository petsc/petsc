#!/bin/bash

# Checks for compliance with 
# Rule: 'Don't use C++ comments in C sources'


# Steps:
# - exclude src/docs/ holding the documentation only
# - run checker script for each file


find src/ include/ -name *.[ch] \
 | grep -v 'src/docs' \
 | xargs $(dirname $0)/cpp-comments.sh
