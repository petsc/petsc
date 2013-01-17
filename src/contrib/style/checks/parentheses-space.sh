#!/bin/bash

# Checks for compliance with 
# Rule: 'No space after a '(' or before a ')''

# Steps:
# - exclude src/docs/ holding the documentation only
# - find lines with '( ' or ' )'
# 



find src/ -name *.[ch] \
 | grep -v 'src/docs' \
 | xargs grep "( \| )"

