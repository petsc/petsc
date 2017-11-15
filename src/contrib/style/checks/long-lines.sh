#!/bin/bash

# Checks for compliance with 
# Rule: 'Generally we like lines less than 150 characters wide. (checks for line longer than 250 characters)'

# Steps:
# - find lines with 250 chars or more
# 

grep -n -H "^.\{250\}" "$@"

