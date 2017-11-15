#!/bin/bash

# Checks for compliance with 
# Rule: 'No space before or after a , in lists'

# Steps:
# - exclude src/docs/ holding the documentation only
# - exclude automatic FORTRAN stuff
# - call checker script
# 

find src/ -name *.[ch] -or -name *.cu \
 | grep -v 'ftn-auto' \
 | xargs $(dirname $0)/space-in-lists.sh

