#!/bin/bash

# Checks for compliance with 
# Rule: 'Indentation for if statements must be done as 'if ('. Similarly for, while, switch.'
#
# if (...) {
#  ...
# } else {
#  ...
# };
#
# Only checks the first line containing the 'if'
#

# Steps:
# - get all lines with 'if(', 'for(', 'while (', 'switch('


grep -n -H " if(\| for(\| while(\| switch(" "$@"


