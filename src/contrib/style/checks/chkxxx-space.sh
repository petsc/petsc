#!/bin/bash

# Checks for compliance with 
# Rule: 'No space before the CHKXXX()'

# Steps:
# - exclude src/docs/ holding the documentation only
# - find lines with '; CHK'
# 



find src/ -name *.[ch] \
 | grep -v 'src/docs' \
 | xargs grep ";\s\s*CHK"

