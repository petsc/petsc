#!/bin/bash

# Checks for compliance with 
# Rule: 'No space between the type and the * in a cast'

# Steps:
# - find lines with ' *)' preceeded by letters and an opening parenthesis
# - ignore uses of PETSC_STDCALL (seems to be legit, but not sure)
# 


grep -n -H "([_a-zA-Z][_a-zA-Z]* \*)" "$@" \
 | grep -v -F "PETSC_STDCALL *)"


