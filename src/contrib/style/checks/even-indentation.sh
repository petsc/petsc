#!/bin/bash

# Checks for compliance with 
# Rule: 'All PETSc function bodies are indented two blanks. Each additional level of loops, if-statements, etc. is indented two more characters. (Reports lines starting with an odd number of blanks, ignoring comments)'

# Steps:
# - Traverse all files in src/
# - exclude src/docs/ holding the documentation only as well as automatic Fortran stuff
# - Strip off comments using GCC
# - find lines starting with an odd number of blanks (this is an obvious violation to the two-blanks-per-level-of-indentation rule)
# 

for f in `find src/ -name *.[ch] -or -name *.cu`
do
 isvalid=`echo "$f" | grep -v "/ftn-auto/\|src/docs/";`
 if [ -n "$isvalid" ]; then
   #echo "$f"
   output=`gcc -fpreprocessed -dD -E -w -x c++ $f | grep "^\s\(\s\s\)*[^ ]"`
   if [ -n "$output" ]; then echo "$f: $output"; fi;
 fi
done

