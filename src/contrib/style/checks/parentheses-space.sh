#!/bin/bash

# Checks for compliance with 
# Rule: 'No space after a '(' or before a ')''

# Steps:
# - exclude src/docs/ holding the documentation only
# - find lines with '( ' or ' )'
# 



find src/ -name *.[ch] -or -name *.cu \
 | grep -v 'src/docs' \
 | xargs grep "( \| )"

