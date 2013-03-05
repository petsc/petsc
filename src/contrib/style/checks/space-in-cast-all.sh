#!/bin/bash

# Checks for compliance with 
# Rule: 'No space between the type and the * in a cast'

# Steps:
# - exclude src/docs/ holding the documentation only
# - exclude automatic FORTRAN stuff
# - call checker script
# 


find src/ -name *.[ch] -or -name *.cu \
 | grep -v 'src/docs' \
 | grep -v 'ftn-auto' \
 | xargs $(dirname $0)/space-in-cast.sh

