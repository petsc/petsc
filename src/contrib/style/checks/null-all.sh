#!/bin/bash

# Checks for compliance with 
# Rule: 'Use NULL instead of PETSC_NULL (which is deprecated)'


# Steps:
# - exclude src/docs/ holding the documentation only
# - ignore FORTRAN stubs
# - ignore petscsys.h because the uses in there are valid
# - call checker script


find src/ include/ -name *.[ch] \
 | grep -v 'src/docs' \
 | grep -v 'ftn-auto' \
 | grep -v 'ftn-custom' \
 | grep -v 'f90-custom' \
 | grep -v 'petscsys.h' \
 | xargs $(dirname $0)/null.sh

