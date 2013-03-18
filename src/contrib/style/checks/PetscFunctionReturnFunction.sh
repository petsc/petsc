#!/bin/bash

# Checks for compliance with 
# Rule: 'No function calls inside PetscFunctionReturn(...);'

# Steps:
# - extract all lines with 'PetscFunctionReturn(' with additional parenthesis following. Additional parenthesis require a character in front in order to qualify as a function call rather than an arithmetic expression.
# - ignore casts such as PetscFunctionReturn((PetscMPIInt)n);

grep -n -H 'PetscFunctionReturn(.*[a-zA-Z](.*).*);' "$@" \
 | grep -v 'PetscFunctionReturn((' 

