#!/bin/bash

# Checks for compliance with 
# Rule: 'No space after a '(' or before a ')''

# Steps:
# - exclude src/docs/ holding the documentation only
# - exclude automatic Fortran stubs
# - exclude a few files containing corner cases
# - find lines with '( ' or ' )'
# - Remove things inside sowing comments (lines starting with $)
# - Ignore parentheses inside comments
# - Ignore special case inside printf() with %g and %d



find src/ -name *.[ch] -or -name *.cu \
 | grep -v 'src/docs' \
 | grep -v '/ftn-auto/' \
 | grep -v '/da/dalocal.c' \
 | grep -v '/da/dagetelem.c' \
 | grep -v '/dm/examples/tests/ex36.c' \
 | grep -v '/ts/examples/tutorials/phasefield/biharmonic.c' \
 | grep -v '/snes/examples/tutorials/ex4.c' \
 | grep -v '/sys/python/pythonsys.c' \
 | grep -v '/mat/impls/aij/.*aij.c' \
 | xargs grep "( \| )" \
 | grep -v -F ":$" \
 | grep -v "/\*.*( " \
 | grep -v "[pP]rintf(.*%[dg]"

