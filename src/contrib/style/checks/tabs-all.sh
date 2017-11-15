#!/bin/bash

# Checks for compliance with 
# Rule: 'No tabs are allowed in any of the source code.'
#
# Steps:
# - Check src/ folder for tabs ('\t') in any source files
# - Exclude docs/ folder
# - Ignore Fortran stubs
# - Ignore output folder for tutorials/tests/
# - Ignore output files from style reporter (otherwise feedback loop...)
# - Run checker script on each file


find src/ -name *.[ch] \
 | grep -v -F ".html" \
 | grep -v -F "src/docs/" \
 | grep -v -F "/ftn-auto/" \
 | grep -v -F "src/contrib/style/html" \
 | xargs $(dirname $0)/tabs.sh


