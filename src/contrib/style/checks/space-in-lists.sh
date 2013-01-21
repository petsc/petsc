#!/bin/bash

# Checks for compliance with 
# Rule: 'No space before or after a , in lists'

# Steps:
# - exclude src/docs/ holding the documentation only
# - exclude automatic FORTRAN stuff
# - find lines with ' ,' or ', '
# 



for f in `find src/ -name *.[ch] -or -name *.cu`
do
 isvalid=`echo "$f" | grep -v "/ftn-auto/\|src/docs/";`
 if [ -n "$isvalid" ]; then
   #echo "$f"
   output=`gcc -fpreprocessed -dD -E -w -x c++ $f | grep " ,\|, "`
   if [ -n "$output" ]; then echo "$f: $output"; fi;
 fi
done


