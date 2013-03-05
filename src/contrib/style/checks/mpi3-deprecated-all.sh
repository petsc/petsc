#!/bin/bash

# Checks for compliance with 
# Rule: 'Deprecated MPI routines require special care.'
# See http://www.mpi-forum.org/docs/mpi-3.0/mpi30-report.pdf for hints on fixing the issues.

# Steps:
# - exclude src/docs/ holding the documentation only
# - exclude automatic FORTRAN stuff
# - run checker script
# 

find src/ -name *.[ch] -or -name *.cu | xargs $(dirname $0)/mpi3-deprecated.sh

