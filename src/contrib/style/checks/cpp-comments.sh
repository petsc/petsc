#!/bin/bash

# Checks for compliance with 
# Rule: 'Don't use C++ comments in C sources'


# Steps:
# - exclude src/docs/ holding the documentation only
# - find lines with //
# - Remove lines containing http, ftp: or file: as in http:// or https://
# - Remove C++ comments inside C comments escaped with a Dollar-sign (used for sowing)
# - Remove HTML doctype declaration
# - Ignore Sieve in dm/impls/mesh/
# - Ignore a special string inside src/snes/impls/test/snestest.c



find src/ include/ -name *.[ch] \
 | grep -v 'src/docs' \
 | xargs grep "//" \
 | grep -v "http\|ftp:\|file:\|://" \
 | grep -v -F ":$" \
 | grep -v -F "<!DOCTYPE" \
 | grep -v -F "src/dm/impls/mesh/" \
 | grep -v -F "||//J||"

