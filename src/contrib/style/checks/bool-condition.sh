#!/bin/bash

# Checks for compliance with 
# Rule: 'Do not use if (rank == 0) or if (v == PETSC_NULL) or if (flg == PETSC_TRUE) or if (flg == PETSC_FALSE)'
# Note: Comparison with PetscScalar should still be (val == 0) in order to be compatible with complex.

# Steps:
# - exclude src/docs/ holding the documentation only
# - exclude automatic FORTRAN stuff
# - find lines with the patterns in rule. 
# 



find src/ -name *.[ch] -or -name *.cu \
 | grep -v 'src/docs' \
 | grep -v 'ftn-auto' \
 | xargs grep "==\s*0)\|(v\s*==\s*PETSC_NULL)\|==\s*PETSC_TRUE)\|==\s*PETSC_FALSE)"

