#!/bin/bash

# Checks for compliance with 
# Rule: 'Use PetscMalloc(), PetscNew(), PetscFree() instead of malloc(), free(), whenever possible'


# Steps:
# - find any malloc() and free() calls


grep -n -H "[ =)]malloc(\|[ =)]free(" "$@"



