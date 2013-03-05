#!/bin/bash

# Checks for compliance with 
# Rule: 'No function calls inside PetscFunctionReturn(...);'

# Steps:
# - exclude src/docs/ holding the documentation only
# - call checker script

find src/ -name *.[ch] -or -name *.cu \
 | grep -v 'src/docs' \
 | xargs $(dirname $0)/PetscFunctionReturnFunction.sh

