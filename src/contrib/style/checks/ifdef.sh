#!/bin/bash

# Checks for compliance with 
# Rule: 'Do not use #ifdef or #ifndef rather use #if defined(...) or #if !defined(...)'

# Steps:
# - exclude src/docs/ holding the documentation only
# - exclude automatic FORTRAN stuff
# - find lines with '#ifdef' of '#ifndef'
# 



find src/ -name *.[ch] \
 | grep -v 'src/docs' \
 | grep -v 'ftn-auto' \
 | xargs grep "#ifn*def"

