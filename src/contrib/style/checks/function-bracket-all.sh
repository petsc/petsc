#!/bin/bash

# Checks for compliance with 
# Rule: 'In function declaration the open bracket { should be on the next line, not on the same line as the function name and arguments'

# Steps:
# - exclude src/docs/ holding the documentation only, and ftn-auto directories
# - call checker script

find src/ -name *.[ch] -or -name *.cu \
 | grep -v 'src/docs' \
 | grep -v 'ftn-auto' \
 | xargs grep $(dirname $0)/function-bracket-all.sh

