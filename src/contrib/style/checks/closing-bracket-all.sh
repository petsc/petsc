#!/bin/bash

# Checks for compliance with 
# Rule: 'The closing bracket } should always be on its own line' (if, for, while, etc.)

# Steps:
# - exclude src/docs/ holding the documentation only, and ftn-auto directories
# - call checker script

find src/ -name *.[ch] -or -name *.cu \
 | grep -v 'src/docs' \
 | grep -v 'ftn-auto' \
 | xargs $(dirname $0)/closing-bracket.sh

