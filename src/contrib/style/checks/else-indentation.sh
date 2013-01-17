#!/bin/bash

# Checks for compliance with 
# Rule: 'Indentation for else within if statements must be done as as '} else {''
#
# Only checks the line with 'else'
#

# Steps:
# - exclude src/docs/ holding the documentation only, and ftn-auto directories
# - get all lines with 'else' followed by a space or a curly brace
# - remove all good uses of '} else {'
# - remove all good uses of '} else if (...) {'
# - remove else with single command on same line
# - remove preprocessor stuff



find src/ -name *.[ch] \
 | grep -v 'src/docs' \
 | grep -v 'ftn-auto' \
 | xargs grep "else[{\s]*" \
 | grep -v "} else {" \
 | grep -v "else if (.*) {" \
 | grep -v " else [^{]*;" \
 | grep -v "#else" 

