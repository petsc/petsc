#!/bin/bash

# Checks for compliance with 
# Rule: 'No space before the CHKXXX()'

# Steps:
# - exclude src/docs/ holding the documentation only
# - call checker script
# 



find src/ -name *.[ch] -or -name *.cu \
 | grep -v 'src/docs' \
 | xargs $(dirname $0)/chkxxx-space.sh

