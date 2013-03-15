#!/bin/bash

# Checks for compliance with 
# Rule: 'Do not use #ifdef or #ifndef rather use #if defined(...) or #if !defined(...)'

# Steps:
# - find lines with '#ifdef' or '#ifndef'
# 

grep -n -H "#ifn*def" "$@"

