#!/bin/bash

# Checks for compliance with 
# Rule: 'All PETSc function bodies are indented two blanks. Each additional level of loops, if-statements, etc. is indented two more characters. (Reports lines starting with an odd number of blanks, ignoring comments)'

# Steps:
# - Strip off comments using GCC
# - find lines starting with an odd number of blanks (this is an obvious violation to the two-blanks-per-level-of-indentation rule)


for f in "$@"
do
 output=`gcc -fpreprocessed -dD -E -w -x c++ $f | grep "^\s\(\s\s\)*[^ ]"`
 if [ -n "$output" ]; then
   #numlines=`echo "$output" | wc -l`
   #echo "$f: $numlines";
   echo "$f: $output";
 fi
done

