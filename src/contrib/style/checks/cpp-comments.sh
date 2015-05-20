#!/bin/bash

# Checks for compliance with 
# Rule: 'Don't use C++ comments in C sources'


# Steps:
# - find lines with //
# - Remove lines containing http, ftp: or file: as in http:// or https://
# - Remove C++ comments inside C comments escaped with a Dollar-sign (used for sowing)
# - Remove HTML doctype declaration
# - Ignore a special string inside src/snes/impls/test/snestest.c
# - Ignore other special cases

grep -n -H -F "//" "$@" \
 | grep -v "http\|ftp:\|file:" \
 | grep -v -F ":$" \
 | grep -v -F "<!DOCTYPE" \
 | grep -v -F "||//J||" \
 | grep -v -F "\"://\""

