#!/bin/bash

# Checks for compliance with 
# Rule: 'Lines with ierr = ... should end with CHKERRQ();'


# Steps:
# - exclude src/docs/ holding the documentation only
# - ignore FORTRAN stubs
# - call checker script for all other files 


find src/ include/ -name *.[ch] \
 | grep -v 'src/docs' \
 | grep -v 'ftn-auto' \
 | grep -v 'ftn-custom' \
 | grep -v 'f90-custom' \
 | xargs $(dirname $0)/ierr-chkerr.sh


