#!/bin/bash

# Checks for compliance with 
# Rule: 'Do not leave chunks of commented out code in the source files (checks for #if 0)'


# Steps:
# - exclude src/docs/ holding the documentation only
# - ignore FORTRAN stubs


find src/ include/ -name *.[ch] \
 | grep -v 'src/docs' \
 | grep -v 'ftn-auto' \
 | grep -v 'ftn-custom' \
 | grep -v 'f90-custom' \
 | xargs $(dirname $0)/if0.sh



