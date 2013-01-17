#!/bin/bash

# Checks for compliance with 
# Rule: 'No tabs are allowed in any of the source code.'
#
# Steps:
# - Check src/ folder for tabs ('\t') in any files except Makefiles, generated HTML files, and LaTeX-manuals
#

find src/ \
 | grep -v akefile \
 | grep -v -F ".html" \
 | grep -v -F "src/docs/" \
 | grep -v -F "src/contrib/style/html" \
 | xargs grep -P '\t'


