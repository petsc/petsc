#!/bin/bash

# Checks for compliance with 
# Rule: 'Use NULL instead of PETSC_NULL (which is deprecated)'


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
 | xargs grep "PETSC_NULL" \
 | grep -v "PETSC_NULL_" \
 | grep -v "petscsys.h:"

