#!/bin/bash

# Checks for compliance with 
# Rule: 'No space after the ) in a cast'

# Steps:
# - strip off comments using GCC
# - get all lines with '(...) ' not followed by any of the characters {|&<>, where only letters, underscore, and '*' are allowed inside the parentheses. At least two characters need to be inside the parentheses
# - Throw away all cases of type 'if (flag) ...;'
# - Ignore macro definitions


for f in "$@"
do
 isvalid=`echo "$f" | grep -v "/ftn-auto/\|src/docs/";`
 if [ -n "$isvalid" ]; then
   #echo "$f"
   output=`gcc -fpreprocessed -dD -E -w -x c++ $f | grep "([_a-zA-Z][_a-zA-Z][_a-zA-Z*]*)\s\s*[^{|&<>]" | grep -v "if ([_a-zA-Z*]*)\s\s*[^{|&<>]" | grep -v "#define"`
   if [ -n "$output" ]; then echo "$f: $output"; fi;
 fi
done



