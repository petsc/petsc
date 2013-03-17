#!/bin/bash

# Checks for compliance with 
# Rule: 'Rules on if, for, while, etc. imply that there is no '){' allowed.'
#

# Steps:
# - get all lines with '){'

grep -n -H "){" "$@"

