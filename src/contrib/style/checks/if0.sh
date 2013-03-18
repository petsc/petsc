#!/bin/bash

# Checks for compliance with 
# Rule: 'Do not leave chunks of commented out code in the source files (checks for #if 0)'


# Steps:
# - exclude src/docs/ holding the documentation only
# - ignore FORTRAN stubs
# - find any use of '#if 0' and variations thereof


grep -n -H "#if\s*0" "$@"



