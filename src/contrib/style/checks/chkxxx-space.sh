#!/bin/bash

# Checks for compliance with 
# Rule: 'No space before the CHKXXX()'

# Steps:
# - find lines with one or more spaces between the semicolon and 'CHK'
# 

grep -n -H ";\s\s*CHK" "$@"

