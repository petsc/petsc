#!/bin/bash

# Checks for compliance with 
# Rule: 'Indentation for else within if statements must be done as as '} else {''
#
# Only checks the line with 'else'
#

# Steps:
# - only look at sources where comments are stripped out (using gcc for that)
# - get all lines with 'else' followed by a space or a curly brace
# - remove all good uses of '} else {'
# - remove all good uses of 'else {' preceeded by blanks only
# - remove all good uses of '} else if (...) {'
# - remove else with single command on same line
# - remove preprocessor stuff


for f in "$@"
do
 output=`gcc -fpreprocessed -dD -E -w -x c++ $f | grep "else[{\s]*" | grep -v "} else {" | grep -v "^\s*else {" | grep -v "}* else if (.*)" | grep -v " else [^{]*;" | grep -v "#\s*else"`
 if [ -n "$output" ]; then echo "$f: $output"; fi
done

