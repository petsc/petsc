#!/bin/bash

# Checks for compliance with 
# Rule: 'We want to get rid of Macros'


# Steps:
# - exclude src/docs/ holding the documentation only
# - call checker script


find src/ include/ -name *.[ch] \
 | grep -v 'src/docs' \
 | grep -v 'ftn-auto' \
 | xargs $(dirname $0)/macros.sh


