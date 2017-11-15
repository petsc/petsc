#!/bin/bash

# Checks for compliance with 
# Rule: 'Do not use #ifdef or #ifndef rather use #if defined(...) or #if !defined(...)'

# Steps:
# - exclude src/docs/ holding the documentation only
# - exclude automatic FORTRAN stuff
# - call checker script for other files
# 



find src/ -name *.[ch] -or -name *.cu \
 | grep -v 'src/docs' \
 | grep -v 'ftn-auto' \
 | xargs $(dirname $0)/ifdef.sh

