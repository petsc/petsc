#!/bin/bash

# Checks for compliance with 
# Rule: 'No space between the type and the * in a cast'

# Steps:
# - exclude src/docs/ holding the documentation only
# - exclude automatic FORTRAN stuff
# - find lines with ' *)' preceeded by letters and an opening parenthesis
# 



find src/ -name *.[ch] -or -name *.cu \
 | grep -v 'src/docs' \
 | grep -v 'ftn-auto' \
 | xargs grep "([_a-zA-Z][_a-zA-Z]* \*)"

