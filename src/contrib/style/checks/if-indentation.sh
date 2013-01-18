#!/bin/bash

# Checks for compliance with 
# Rule: 'Indentation for if statements must be done as 'if (...) {''
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
# - get all lines with 'if' followed by a space or a opening parenthesis
# - keep only lines with a pair of parenthesis in order to exclude 'if' in comments
# - remove valid uses of if (multiline, single-line)
# - remove valid uses of else
# - remove preprocessor stuff
# - remove uses of 'if' inside a word


find src/ -name *.[ch] -or -name *.cu \
 | grep -v 'src/docs' \
 | grep -v 'ftn-auto' \
 | xargs grep "if[(\s]" \
 | grep "(.*)" \
 | grep -v "if\s(.*)\s{[^}]*" \
 | grep -v "if\s(.*)[^{]*;" \
 | grep -v "#.*if" \
 | grep -v "if[a-zA-Z]\|[a-zA-Z]if"

