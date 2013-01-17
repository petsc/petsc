#!/bin/bash

# Checks for compliance with 
# Rule: 'Use PetscFunctionReturn(returnvalue); not return(returnvalue);'

# Steps:
# - exclude src/docs/ holding the documentation only
# - extract all lines with 'PetscFunctionReturn(' with additional parenthesis following. Additional parenthesis require a character in front in order to qualify as a function call rather than an arithmetic expression.
# - ignore casts such as PetscFunctionReturn((PetscMPIInt)n);

find src/ -name *.[ch] \
 | grep -v 'src/docs' \
 | xargs grep 'PetscFunctionReturn(.*[a-zA-Z](.*).*);' \
 | grep -v 'PetscFunctionReturn((' 

