#!/bin/bash

# Checks for compliance with 
# Rule: 'Indentation for if statements must be done as 'if ('. Similarly for, while, switch.'
#
# if (...) {
#  ...
# } else {
#  ...
# };
#
# Only checks the first line containing the 'if'
#

# Steps:
# - exclude src/docs/ holding the documentation only, and ftn-auto directories
# - call checker script


find src/ -name *.[ch] -or -name *.cu \
 | grep -v 'src/docs' \
 | grep -v 'ftn-auto' \
 | xargs $(dirname $0)/if-indentation.sh


