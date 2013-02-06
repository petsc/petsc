#!/bin/bash

# Checks for compliance with 
# Rule: 'Lines with ierr = ... should end with CHKERRQ();'


# Steps:
# - exclude src/docs/ holding the documentation only
# - ignore FORTRAN stubs
# - find any line with a ierr =
# - ignore lines with CHKERR.*
# - ignore lines ending with ,
# - ignore ierr = PetscFinalize();


find src/ include/ -name *.[ch] \
 | grep -v 'src/docs' \
 | grep -v 'ftn-auto' \
 | grep -v 'ftn-custom' \
 | grep -v 'f90-custom' \
 | xargs grep "ierr *=" \
 | grep -v "CHKERR" \
 | grep -v ",$" \
 | grep -v "ierr = PetscFinalize();" \
 | grep -v -F "*ierr"


