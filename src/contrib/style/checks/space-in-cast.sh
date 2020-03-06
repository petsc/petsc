#!/bin/bash

# Checks for compliance with
# Rule: 'No space between the type and the * in a cast'

# Steps:
# - find lines with ' *)' preceeded by letters and an opening parenthesis
#


grep -n -H "([_a-zA-Z][_a-zA-Z]* \*)" "$@"


