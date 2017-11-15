#!/bin/bash

# Checks for compliance with 
# Rule: 'Do not put a blank line immediately before PetscFunctionReturn;' <br />(reports filenames only)

# Steps:
# - Run custom python script on each file in src/ tree

find src/ -name *.[ch] -or -name *.cu \
 | xargs python $(dirname $0)/PetscFunctionReturn.py

