#!/bin/bash

# Checks for compliance with 
# Rule: 'There should be a blank line before PetscFunctionBegin;' <br />(reports filenames only)

# Steps:
# - Run custom python script on each file in src/ tree

find src/ -name *.[ch] -or -name *.cu \
 | xargs python src/contrib/style/checks/PetscFunctionBegin2.py


