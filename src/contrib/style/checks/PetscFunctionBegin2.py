#!/usr/bin/env python

# Checks for compliance with the rule 'There should be a blank line before PetscFunctionBegin;' 
# (note that this is implicit from the rule 'There must be a single blank line between the local variable declarations and the body of the function.'
#  and 'The first line of the executable statments must be PetscFunctionBegin;')


# Steps:
# - Read each file argument to a single string, then run a regular expression on it.
#
# Note: Only file name is printed on match, no details
#


import sys
import re



for arg in sys.argv[1:]:
    inputfile = open(arg, "r")
    filestring = inputfile.read()
    inputfile.close()

    if re.search("\s*PetscFunctionBegin;\s*\n\s*\n", filestring):
        print arg

    if re.search("\n\s*[^ ]\s*\n\s*PetscFunctionBegin", filestring):
        print arg



