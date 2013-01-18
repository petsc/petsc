#!/bin/bash

# Checks for compliance with 
# Rule: 'No space before the CHKXXX()'

# Steps:
# - exclude src/docs/ holding the documentation only
# - find lines with '; CHK'
# 



find src/ -name *.[ch] -or -name *.cu \
 | grep -v 'src/docs' \
 | xargs grep ";\s\s*CHK"

