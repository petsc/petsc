#!/bin/bash

# Checks for compliance with 
# Rule: 'All PETSc function bodies are indented two blanks. Each additional level of loops, if-statements, etc. is indented two more characters. (Reports lines starting with an odd number of blanks, ignoring comments)'

# Steps:
# - Traverse all files in src/
# - exclude src/docs/ holding the documentation only as well as automatic Fortran stuff
# - Run script checking the individual files
 

find src/ include/ -name *.[ch] -or -name *.cu \
 | grep -v '/ftn-auto/\|src/docs/' \
 | xargs $(dirname $0)/even-indentation.sh

