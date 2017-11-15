#!/bin/bash

# Checks for compliance with 
# Rule: 'No space after the ) in a cast'

# Steps:
# - go through all files in src/
# - exclude src/docs/ holding the documentation only, and ftn-auto directories
# - Run checker script


find src/ -name *.[ch] -or -name *.cu \
 | grep -v 'src/docs' \
 | xargs $(dirname $0)/space-after-cast.sh



