#!/bin/bash

# Checks for compliance with 
# Rule: 'No space after a '(' or before a ')''

# Steps:
# - find lines with '( ' or ' )'
# - Remove things inside sowing comments (lines starting with $)
# - Ignore parentheses inside comments
# - Ignore special case inside printf() with %g and %d


grep -n -H "( \| )" "$@" \
 | grep -v -F ":$" \
 | grep -v "/\*.*( " \
 | grep -v "[pP]rintf(.*%[dg]"

