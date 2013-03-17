#!/bin/bash

# Checks for compliance with 
# Rule: 'No space before or after a , in lists'

# Steps:
# - strip off comments using GCC
# - find lines with ' ,' or ', '
# 


for f in "$@"
do
 isvalid=`echo "$f" | grep -v "/ftn-auto/\|src/docs/";`
 if [ -n "$isvalid" ]; then
   #echo "$f"
   output=`gcc -fpreprocessed -dD -E -w -x c++ $f | grep " ,\|, "`
   if [ -n "$output" ]; then echo "$f: $output"; fi;
 fi
done


