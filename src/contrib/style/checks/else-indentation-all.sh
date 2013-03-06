#!/bin/bash

# Checks for compliance with 
# Rule: 'Indentation for else within if statements must be done as as '} else {''
#
# Only checks the line with 'else'
#

# Steps:
# - exclude src/docs/ holding the documentation only, and ftn-auto directories
# - call checker script for files


find src/ -name *.[ch] -or -name *.cu | xargs $(dirname $0)/else-indentation.sh

