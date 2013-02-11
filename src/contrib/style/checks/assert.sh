#!/bin/bash

# Checks for compliance with 
# Rule: 'Don't use assert().'


# Steps:
# - exclude src/docs/ holding the documentation only
# - find any line with assert


find src/ include/ -name *.[ch] \
 | grep -v 'src/docs' \
 | xargs grep "assert *("

