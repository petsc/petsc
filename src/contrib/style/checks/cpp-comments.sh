#!/bin/bash

# Checks for compliance with 
# Rule: 'Don't use C++ comments in C sources'


# Steps:
# - exclude src/docs/ holding the documentation only
# - find lines with //
# - Remove lines containing http://



find src/ -name *.[ch] \
 | grep -v 'src/docs' \
 | xargs grep "//" \
 | grep -v "http://"

