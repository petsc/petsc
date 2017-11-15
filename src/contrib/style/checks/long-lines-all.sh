#!/bin/bash

# Checks for compliance with 
# Rule: 'Generally we like lines less than 150 characters wide. (checks for line longer than 250 characters)'

# Steps:
# - exclude src/docs/ holding the documentation only
# - exclude automatic FORTRAN stuff
# - call checker script
# 

find src/ -name *.[ch] -or -name *.cu \
 | grep -v 'src/docs' \
 | grep -v 'ftn-auto' \
 | xargs $(dirname $0)/long-lines.sh

