#!/bin/bash

# Checks for compliance with 
# Rule: 'Use PetscMalloc(), PetscNew(), PetscFree() instead of malloc(), free(), whenever possible'


# Steps:
# - exclude src/docs/ holding the documentation only
# - ignore FORTRAN stubs
# - run checker script


find src/ include/ -name *.[ch] \
 | grep -v 'src/docs' \
 | grep -v 'ftn-auto' \
 | grep -v 'ftn-custom' \
 | grep -v 'f90-custom' \
 | xargs  $(dirname $0)/malloc-free.sh



