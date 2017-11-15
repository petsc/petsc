#!/bin/bash

# Checks for compliance with 
# Rule: 'Use NULL instead of PETSC_NULL (which is deprecated)'


# Steps:
# - find lines with PETSC_NULL in them
# - ignore cases where PETSC_NULL is part of a longer preprocessor constant


grep -n -H "PETSC_NULL" "$@" \
 | grep -v "PETSC_NULL_"

