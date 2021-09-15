#!/bin/bash

# Checks for compliance with 
# Rule: 'Do not use if (v == NULL) or if (flg == PETSC_TRUE) or if (flg == PETSC_FALSE)'
# Note: Comparison with PetscScalar should still be (val == 0) in order to be compatible with complex.

# Steps:
# - exclude src/docs/ holding the documentation only
# - exclude automatic FORTRAN stuff
# - call checker for each file
# 



find src/ -name *.[ch] -or -name *.cu \
 | grep -v 'src/docs' \
 | grep -v 'ftn-auto' \
 | xargs $(dirname $0)/bool-condition.sh

