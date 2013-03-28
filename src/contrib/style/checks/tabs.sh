#!/bin/bash

# Checks for compliance with 
# Rule: 'No tabs are allowed in any of the source code.'
#
# Steps:
# - find lines with a tab

grep -n -H -E '\t' "$@"


