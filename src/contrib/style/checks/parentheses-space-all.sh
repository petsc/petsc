#!/bin/bash

# Checks for compliance with 
# Rule: 'No space after a '(' or before a ')''

# Steps:
# - exclude src/docs/ holding the documentation only
# - exclude automatic Fortran stubs
# - exclude a few files containing corner cases



find src/ -name *.[ch] -or -name *.cu \
 | grep -v 'src/docs' \
 | grep -v '/ftn-auto/' \
 | grep -v '/da/dalocal.c' \
 | grep -v '/da/dagetelem.c' \
 | grep -v '/dm/tests/ex36.c' \
 | grep -v '/ts/tutorials/phasefield/biharmonic.c' \
 | grep -v '/snes/tutorials/ex4.c' \
 | grep -v '/sys/python/pythonsys.c' \
 | grep -v '/mat/impls/aij/.*aij.c' \
 | xargs $(dirname $0)/parentheses-space.sh

